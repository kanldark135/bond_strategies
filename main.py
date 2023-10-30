#%%

''' 추가 과제
1) (특히 중요) 금리데이터 단말기 말고 웹 등으로 확보되게끔 하는 과정. 경제지표 관련은 한은 API 등등 사용하면 될듯
아래 코드는 외부에서 자동으로 끌어올 수 있게 되면 끌어오는 데이터 포맷에 맞게 전부 변경'''
 
import pandas as pd
import numpy as np
import quant
import func

#1. 데이터 수집 / 확보

# 이미 외부에서 가져옴 -> 단말기에서 가져오는 데이터 API 등을 이용하여 아예 서버로 끌어올수 있는지 문의

df_idx = pd.read_excel("./bond_momentum_값복사.xlsx", sheet_name = "raw_data_infomax", skiprows = [0, 1], usecols = "A:O")
df_raw_data = pd.read_excel("./bond_momentum_값복사.xlsx", sheet_name = "raw_data_infomax", skiprows = [0, 1], usecols = "P:W")

df_idx_column_names = ['date',
                "idx_agg_23", 
                "dur_agg_23", 
                "ytm_agg_23", 
                "idx_tre_510", 
                "dur_tre_510", 
                "ytm_tre_510", 
                "idx_fin_23", 
                "dur_fin_23", 
                "ytm_fin_23",
                "idx_safe_23", 
                "dur_safe_23", 
                "ytm_safe_23",
                "idx_call_1d", 
                "ytm_call_1d"
                ]

df_data_column_names = ["tre_2y", 
                "tre_3y", 
                "tre_5y", 
                "tre_10y",
                "fin_2y", 
                "fin_3y",
                "kr_credit_risk",
                "kr_CLI"
                ]

# 인포맥스 구성이 2행 형태라 어쩔 수 없이수기로 1행짜리 컬럼 재구성

df_idx.columns = df_idx_column_names
df_raw_data.columns = df_data_column_names

df_idx.set_index('date', inplace = True)
df_raw_data.index = df_idx.index

''' 수정할거 1. 데이터 수집 인포맥스에서 끌어오는 형태에서 -> 자동화되게끔 해야 '''

df_raw_data = df_raw_data.loc[~(df_raw_data.index == "조회요청실패")] # 인포맥스에서 가져오는 경우 값복사해서 안 가져오면 첫 행에서 발생
df_idx = df_idx.loc[~(df_idx.index == "조회요청실패")] # 인포맥스에서 가져오는 경우 값복사해서 안 가져오면 첫 행에서 발생

df_raw_data.index = pd.to_datetime(df_raw_data.index)
df_idx.index = pd.to_datetime(df_idx.index)

# 사용하는 만기구간 컬럼명 -> 전략 수정시 같이 수정


#%% 데이터 가공 

def momentum_score(df, weight_list = [1, 3, 6, 12]):
    df_res = df.copy()
    df_res.iloc[:] = 0

    for i in range(len(weight_list)): # type(i) == integer
        ma = weight_list[-i - 1] * 100 / df.rolling(window = 30 * weight_list[i], min_periods = 1).mean()
        df_res = df_res + ma

    res = df_res / sum(weight_list)
    return res

def get_weights(df_raw_data):

    # 듀레이션 모멘텀
    df_raw_data['treasury_index'] = df_raw_data.loc[:, 'tre_2y': 'tre_10y'].mean(axis = 1)
    df_score_daily = momentum_score(df_raw_data['treasury_index'])\
        .to_frame()\
        .rename(columns = {'treasury_index' : 'treasury_momentum'})

    # 커브 모멘텀 : 일단 패스

    # 크레딧 모멘텀
    df_raw_data['credit_spread'] =  df_raw_data['fin_2y'] - df_raw_data['tre_2y']
    df_score_daily['credit_momentum'] = momentum_score(df_raw_data['credit_spread'])


    ## 일자료 -> 월자료화하기
    df_score_monthly = func.extract_last_price(df_score_daily, interval = "M")
    df_score_monthly['treasury_momentum'] = df_score_monthly['treasury_momentum'].rolling(window = 5, min_periods = 1).mean()
    df_score_monthly = df_score_monthly.merge(df_raw_data['kr_CLI'], how = "left", left_index = True, right_index = True)

    # CLI 모멘텀 및 여타 모멘텀 스코어의 절대모멘텀 산출
    df_chg = df_score_monthly.pct_change(1)
    df_score_monthly['duration_ow'] = np.where(np.logical_and(df_chg['treasury_momentum']> 0, df_chg['kr_CLI'] < 0), 1, 0)
    # 커브
    df_score_monthly['credit_ow'] = np.where(df_chg['credit_momentum'] > 0, 1, 0)
    df_score_monthly['long_duration'] = np.where(df_score_monthly['duration_ow'] == 1, 0.5, 0)
    df_score_monthly['long_credit'] = np.where(df_score_monthly['credit_ow'] == 1, 1 - df_score_monthly['long_duration'], 0.5 - df_score_monthly['long_duration'])
    # 커브
    df_score_monthly['safe_asset'] = 1 - df_score_monthly['long_duration'] - df_score_monthly['long_credit']

    return df_score_monthly

#%%   비중배분전략 구현 (backtest 및 현재 비중 도출)

def get_results(df_score_monthly, start_date = start_date, portfolio_type = portfolio_type):

    '''
    portfolio_type : 각각 'aggressive', 'semi_aggressive', 'netural', 'semi_stable', 'stable', 'ultra_stable'
    '''

    dict_portfolio_type = {
    'aggressive' : 1,
    'semi_aggressive' : 0.8,
    'neutral' : 0.6,
    'semi_stable' : 0.4,
    'stable' : 0,
    'ultra_stable' : np.nan # 아예 여기 유니버스랑 상관없는 1년짜리 (위험등급 6짜리) 채권 ETF 매수
    }

    multiplier = dict_portfolio_type.get(portfolio_type) 

    df_res_monthly = df_score_monthly[['long_duration', 'long_credit', 'safe_asset']]
    df_res_monthly = df_res_monthly * multiplier 
    df_res_monthly['safe_asset'] = df_res_monthly['safe_asset'] + (1 - multiplier) # safe_asset 비중 강제로 추가

    # 일비중으로 치환
    df_res_monthly.loc[pd.to_datetime(start_date) - pd.DateOffset(days = 1)] = [0, 0, 1]
    df_res_monthly.sort_index(inplace = True)
    df_res_daily = df_res_monthly.resample('1D').ffill().shift(1).iloc[1:]

    return {'df_res_monthly' : df_res_monthly,
            'df_res_daily' :  df_res_daily
    }

# 2. 팩터별 수익률 계산
def get_returns(df_idx, asset_list = asset_list, start_date = start_date):

    'df_idx : 채권지수 누적값 (=누적수익률 df_cumret_daily 와 같음)'
    'asset_list : 임의로 정의한 실제 필요한 채권지수의 이름. 위에서 정의'
    # df_cumret_daily
    df_cumret_daily = quant.df_cumret(df_idx.filter(regex = "idx_*")) # 채권수익률 지수만 추림 (regex = idx)
    df_cumret_daily = df_cumret_daily[[asset_list.get('duration'), asset_list.get('credit'), asset_list.get('safe')]] # 채권수익률 지수만 추림 (regex = idx)
    
    # df_ret_daily
    df_ret_daily = (df_cumret_daily + 1).pct_change(1)
    df_ret_daily.iloc[0] = 0
    
    # df_cumret_monthly
    df_cumret_monthly = df_cumret_daily.loc[df_score_monthly.index]
    df_cumret_monthly.loc[pd.to_datetime(start_date)] = 0 # 매달 말 누적수익률
    df_cumret_monthly.sort_index(inplace = True)

    # df_ret_monthly
    df_ret_monthly = (df_cumret_monthly + 1).pct_change(1) # 매달 말 / 전달 말로 월간 수익률 산출
    df_ret_monthly.iloc[0] = 0

    return {'df_cumret_daily' : df_cumret_daily,
            'df_ret_daily' : df_ret_daily,
            'df_cumret_monthly' : df_cumret_monthly,
            'df_ret_monthly' : df_ret_monthly
    }

# 3. Transaction Cost 반영
def apply_tc(df, tc = 0.0005): #5bp 가정

    dummy = df.shift(1)
    df_trade_bool = df.eq(dummy).all(axis = 1)
    df['trade_true'] = df_trade_bool
    df['tc'] = np.where(df['trade_true'], 0, -tc)
    df['tc'].iloc[0] = 0

    return df

# 4. 최종적으로 팩터별 비중 X 팩터별 수익률 해서 전략의 수익률 산출 함수
def calc_strat_ret(df_ret, df_result, tc = False):

    shifted_weight = df_result.shift(1)
    shifted_weight.iloc[0] = {'long_duration' : 0, 'long_credit' : 0, 'safe_asset' : 1} # 첫행 na값 조정
    try :
        asset_ret = shifted_weight * df_ret.values
        strat_ret = asset_ret.sum(axis = 1)
    except TypeError:
        asset_ret = shifted_weight.iloc[1:] * df_ret.values
        strat_ret = asset_ret.sum(axis = 1)

    if tc == True:
        df_result = df_result.pipe(apply_tc, tc = 0.0005)
        strat_ret = strat_ret + df_result['tc']
    else:
        pass

    return strat_ret

# %% 지수비중에서 종목비중으로 치환하는 모듈

def get_multiIndex(universe_dict):
    dummy = []
    for key, value in universe_dict.items():
        for i in value:
            dummy.append([key, i])
    res = pd.MultiIndex.from_tuples(dummy)
    return res

def allocate_weight_to_sec(idx_weight, sec_weight, universe_weight):
    for keys in universe:
        sec_weight[keys] = idx_weight[keys]
        sec_weight[keys] = sec_weight[keys].multiply(universe_weight[keys], axis = 1)
    return sec_weight

# %% summary

if __name__ == "__main__":

    # 사전 변수 정의

    start_date = df_idx.index[0] # 마지막날

    asset_list = {'duration' : 'idx_tre_510',
                'curve' : None, # 나중에 전략 보완시 curve 도 추가할수도 있음
                'credit' : 'idx_fin_23',
                'safe' : 'idx_safe_23'} # idx_call_1d 로 switching 가능.
    
    portfolio_type = 'stable'

    universe = {
        'long_duration' : ['KOSEF 국고채10년', # 9년 
                        'TIGER 중장기국채'], # 5년

        'long_credit' : ['KBSTAR 25-11 회사채(AA-이상)액티브', 
                        '히어로즈 26-09 회사채(AA-이상)액티브'],
        
        'safe_asset' : [ # '마이티 26-09 특수채(AAA)액티브', # 제일 이상적 / 거래가 안 됨,
                'KBSTAR 단기국공채액티브', # 1년
                'ACE 중장기국공채액티브', # 5년 수준
                #    'TIGER 23-12 국공채액티브', 
                #    'WOORI 단기국공채액티브',  # 0.5년
                #    'KBSTAR 중장기국공채액티브' # 국고채밖에 없음 X
                ]
    }

    # 위에 universe 에 해당하는 각 종목들의 팩터 내에서의 비중 : 종목비중 변화시 사용
    universe_weight = {
        'long_duration' : [0.5,
                        0.5
                        ],

        'long_credit' : [0.5, 
                        0.5
                        ],
        
        'safe_asset' : [0.5,
                        0.5
                        ]
}
    
    df_score_monthly = get_weights(df_raw_data)

    returns = get_returns(df_idx, asset_list = asset_list, start_date = start_date)
    results = get_results(df_score_monthly, start_date = start_date, portfolio_type = portfolio_type)

    # monthly

    ret_bm = returns['df_ret_monthly'][asset_list.get('safe')]
    ret = calc_strat_ret(returns['df_ret_monthly'], results['df_res_monthly'], tc = False)
    ret_tc = calc_strat_ret(returns['df_ret_monthly'], results['df_res_monthly'], tc = True)

    # daily

    ret_d_bm = returns['df_ret_daily'][asset_list.get('safe')]
    ret_d = calc_strat_ret(returns['df_ret_daily'], results['df_res_daily'], tc = False)
    ret_d_tc = calc_strat_ret(returns['df_ret_daily'], results['df_res_daily'], tc = True)

# -------------------------------------

    # 성과 : 월기준

    strat_monthly = quant.summary(ret_tc, ret_bm, interval = 'M', rf = 0, is_ret = True)
    bm_monthly = quant.summary(ret_bm, ret_bm, interval = 'M', rf = 0, is_ret = True)

    # 성과 : 일기준

    strat_daily = quant.summary(ret_d_tc, ret_d_bm, interval = 'D', rf = 0, is_ret = True)
    bm_daily = quant.summary(ret_d_bm, ret_d_bm, interval = 'D', rf = 0, is_ret = True)

    # 연도별 수익률

    strat_yearly = quant.period_return(ret_tc, is_ret = True)
    bm_yearly = quant.period_return(ret_bm, is_ret = True)

# ----------------------------------------
    ## Final
        # 지수비중에서 종목비중으로 치환
    dummy_weight = pd.DataFrame(columns = get_multiIndex(universe), index = ret.index)

    df_weight = allocate_weight_to_sec(results['df_res_monthly'], dummy_weight, universe_weight)
    weight_this_month = df_weight.iloc[-1]

    
# %% RATB 및 기타 보고용 csv 생성

with pd.ExcelWriter("./ratb_return.xlsx", mode = 'w') as xlsxwriter:

    strat_daily['ret'].to_frame().to_excel(xlsxwriter, sheet_name = 'daily_ret')
    strat_daily['cumret'].to_frame().to_excel(xlsxwriter, sheet_name = 'daily_cumret')

    strat_monthly['ret'].to_frame().to_excel(xlsxwriter, sheet_name = 'monthly_ret')
    strat_monthly['cumret'].to_frame().to_excel(xlsxwriter, sheet_name = 'monthly_cumret')

with pd.ExcelWriter("./ratb_weight.xlsx", mode = 'a', if_sheet_exists = 'replace', engine = 'openpyxl') as appendwriter:

    results['df_res_daily'].to_excel(appendwriter, sheet_name = 'idx_weight')
    