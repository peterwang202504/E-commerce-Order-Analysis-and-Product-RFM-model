
########################################## 資料前處理 ##########################################

'''  說明1-1 取得並讀取kaggle的美國電商銷售訂單資料  '''

#載入套件
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
        
pd.set_option('display.float_format', '{:.2f}'.format)

#讀取檔案
path = kagglehub.dataset_download("ankitbansal06/retail-orders").replace('\\', '/')

data = []
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('csv'):
            data.append(os.path.join(dirname, filename))
        
orders = pd.read_csv(data[0])


'''  說明1-2 快速觀察資料欄位, 屬性, 值, 分布與需要調整的地方  '''

#取得資料的基本資訊
print(orders.head(10))
print(orders.info())
print(orders.describe().T)

orders_object = orders.select_dtypes('object')
for i in orders_object.columns:
    print(orders_object[i].value_counts().head(10))
    print(' ')

order_number = orders.select_dtypes('number')
order_number.hist(bins= 10, figsize = (12, 6))
plt.tight_layout()
plt.show()


'''  說明1-3 為提高資料品質，進行資料處理  '''

#使資料欄位格式一致，提高可讀性
orders.columns = [i.replace(' ', '_').lower() for i in orders.columns]
print(orders.columns)

#使資料欄位屬性正確
orders['order_date'] = pd.to_datetime(orders['order_date'], errors='coerce')
orders['order_id'] = orders['order_id'].astype(str)
orders['postal_code'] = orders['postal_code'].astype(str)
print(orders.dtypes)

#去除5列的異常值
orders = orders[~orders['ship_mode'].isin(['Not Available', 'unknown'])]
print(orders['ship_mode'].value_counts())

#以眾數填充一個缺失值
print(orders.isna().sum().sort_values(ascending = False).head(10))
orders['ship_mode'] = orders['ship_mode'].fillna(orders['ship_mode'].mode()[0])
print(orders[orders['ship_mode'].isna() == True])


'''  說明1-4  
(1)為了確保後續分析的品質，設定並定義欄位，初步驗證資料
(2)2023年美國電商Furniture, Office Supplies, Technology類別的毛利率平均約為32%，淨利率約為6.0%
   故設定報表的折扣率為6.9%，定義profit_rate為淨利率，訂單資料的淨利率為6.0%
'''

#新增欄位與折扣率設定
orders['year'] = orders['order_date'].dt.year
orders['month'] = orders['order_date'].dt.month
orders['day'] = orders['order_date'].dt.day

print(orders['discount_percent'].value_counts()) 
orders['discount_percent'] = orders['discount_percent'] / 100 * 1.97
orders['sell_price'] = orders['list_price'] * (1 - orders['discount_percent'])

orders['revenue'] = orders['sell_price'] * orders['quantity']
orders['cost'] = orders['cost_price'] * orders['quantity']
orders['profit'] = orders['revenue'] - orders['cost']

orders['discount'] = orders['revenue'] * orders['discount_percent']

print(f'每月營收 {orders.groupby("month")["revenue"].sum().astype(int)}')
print(f'淨利率 {(orders["profit"].sum() / orders["revenue"].sum() * 100).round(2)}%') 
print(f'折扣率 {(orders["discount"].sum() / orders["revenue"].sum() * 100).round(2)}%') 
print(f'正負毛比率 {(orders["profit"] > 0).value_counts(normalize = True).round(2) * 100}') 

#orders.to_csv(r'C:/Users/lafk0/Desktop/orders.csv', index=False, encoding='utf-8-sig')


########################################## 函數的設計 ##########################################


'''  設計說明2-1 確定Raw data各欄位都存在的簡易防呆機制  ''' 

def check_columns():
        
    columns_check_object = ['order_id', 'ship_mode', 'segment', 'country', 'city', 
                            'state', 'postal_code', 'region', 'category', 'sub_category', 
                            'product_id']
    columns_check_number = ['cost_price', 'list_price', 'quantity', 'discount_percent', 
                            'year', 'month', 'day', 'sell_price', 'revenue', 'cost', 
                            'profit', 'discount']
    
    for i in columns_check_object:
        if i in table.columns:
            print(table[i].value_counts().head())
        else:
            print(f'{i}欄位不存在，請檢查Raw data')     
        
    for j in columns_check_number:
        if table[j].sum() > 0:
            print(f'{j}  加總{table[j].sum().astype(int)}')
        else:
            print(f'{j}欄位不存在，請檢查Raw data')
            
            
'''  說明2-2 確定各項變數都已宣告的簡易防呆機制  '''

def check_variables():
        
    for i in ['table', 'index_cols', 'cols']:
        try:
            eval(i)
        except NameError:
            print(f'{i}不存在，請設定變數')
            exit()
    print('變數已定義，請執行報表底稿')


'''  說明2-3 建立以總和, 平均, 比率, mom, yoy, diff方式呈現績效指標的報表底稿，以供例行性追蹤  '''

def sales_report(table, index_cols, cols, aggfunc_pick):
        
    #樞紐
    aggfunc = {'revenue' : ['sum'], 
               'profit' : ['sum'],
               'cost' : ['sum'],
               'discount' : ['sum'],
               'quantity' : ['mean'],
               'list_price' : ['mean'],
               'cost_price' : ['mean'],
               'sell_price' : ['mean']}

    table = table.pivot_table(index = index_cols, columns = cols, aggfunc = aggfunc)
    
    #定義後續處理用的欄位名稱
    table_columns_percentage = list(set([(f'{i}_{j}').replace('_sum', "")
                                                     .replace('mean', 'avg') 
                                    for i, j, k in table.columns.to_flat_index()]))
    table_columns_diff = ['revenue', 'profit_rate']

    #讓欄位名稱扁平化，方便後續處理                                                              
    table.columns = ([(f'{i}_{j}_{k}').replace('_sum', "").replace('mean', 'avg') 
                                       for i, j, k in table.columns.to_flat_index()])
   
    #新增常見的績效欄位
    for i in [time_1, time_2]:
        #營收佔比
        table[f'revenue_share_{i}'] = np.where(table[f'revenue_{i}'] != 0, 
                                      (table[f'revenue_{i}'] / table[f'revenue_{i}'].sum()), 0) 
        #淨利率
        table[f'profit_rate_{i}']   = np.where(table[f'revenue_{i}'] != 0, 
                                      (table[f'profit_{i}'] / table[f'revenue_{i}']), 0)
        #折扣率
        table[f'discount_rate_{i}'] = np.where(table[f'revenue_{i}'] != 0, 
                                      (table[f'discount_{i}'] / table[f'revenue_{i}']), 0)
                         
    #新增YoY, MoM欄位
    for i in table_columns_percentage + ['revenue_share', 'profit_rate', 'discount_rate']:
        if cols == 'year':                                                                                                
            table[f'{i}_yoy'] = np.where(table[f'{i}_{time_2}'] != 0, 
                                (table[f'{i}_{time_1}'] - table[f'{i}_{time_2}']) / table[f'{i}_{time_2}'], 0) 
        elif cols == 'month':                                                                                                
            table[f'{i}_mom'] = np.where(table[f'{i}_{time_2}'] != 0, 
                                (table[f'{i}_{time_1}'] - table[f'{i}_{time_2}']) / table[f'{i}_{time_2}'], 0)     
                                                                                                      
    #新增diff(delta)欄位                                                                                                       
    for i in table_columns_diff:
        table[f'{i}_diff'] = table[f'{i}_{time_1}'] - table[f'{i}_{time_2}']
    
    #將欄位名稱調整成常見的格式，增加其可讀性    
    for i in table.columns:
        if 'avg' in i:
            table.rename(columns = {i : f'avg_{i.replace("_avg", "")}'}, inplace = True)
            
    #定義要用的報表欄位，設定排序
    table_columns_name = []
    for i in aggfunc_pick:
        for j in [f'_{time_2}', f'_{time_1}', '_diff', '_yoy', '_mom']:
            if i + j in table.columns:
                table_columns_name.append(i + j)

    sort = []
    for i in aggfunc_pick:
        for j in [f'_{time_1}', '_diff']:
            if i + j in table.columns:
                sort.append(i + j)
                  
    #取出報表
    table = table.loc[:, table_columns_name].reset_index().sort_values(by = sort, ascending = False)
    
    return table


'''  說明2-4 設定KPI條件，產出對應清單，以供瞭解業績的強弱勢點
(1)Star list KPI設定
 1.營收與淨利率的同比, 環比同時上升大於10%
 2.營收的同比, 環比上升大於20%
                        
(2)review list KPI設定
 1.營收與淨利率的同比, 環比同時下降大於10%  
 2.營收的同比, 環比下降大於20%
 3.淨利率的同比, 環比下降大於20%
''' 

def kpi_list(table):
   
    i = ['yoy', 'mom']
    if cols == 'year':
        star_list_kpis_1 = f'revenue_{i[0]} >= 0.1 and profit_rate_{i[0]} >= 0.1'
        star_list_kpis_2 = f'revenue_{i[0]} >= -0.2'

        review_list_kpis_1 = f'revenue_{i[0]} <= -0.1 and profit_rate_{i[0]} <= -0.1'
        review_list_kpis_2 = f'revenue_{i[0]} <= -0.2'
        review_list_kpis_3 = f'profit_rate_{i[0]} <= -0.2'

    elif cols == 'month':
        star_list_kpis_1 = f'revenue_{i[1]} >= 0.1 and profit_rate_{i[1]} >= 0.1'
        star_list_kpis_2 = f'revenue_{i[1]} >= -0.2'

        review_list_kpis_1 = f'revenue_{i[1]} <= -0.1 and profit_rate_{i[1]} <= -0.1'
        review_list_kpis_2 = f'revenue_{i[1]} <= -0.2'
        review_list_kpis_3 = f'profit_rate_{i[1]} <= -0.2'
    
    star_list = table.query(star_list_kpis_1 or star_list_kpis_2)
    review_list = table.query(review_list_kpis_1 or review_list_kpis_2 or review_list_kpis_3)

    return star_list, review_list


'''  說明2-5 透過設定特定欄位int跟其他欄位float + round(2) + %來提高報表的可讀性  '''

def data_types_transform(table): 

    table_columns_name = []
    for i in ['revenue', 'profit', 'cost', 'discount', 'avg_list_price',  'avg_cost_price', 'avg_sell_price']:
        for j in [f'_{time_2}', f'_{time_1}', '_diff']:
            table_columns_name.append(i + j)
    
    for i in table_columns_name:
        if i in table.columns:
            table[i] = table[i].astype(int)
        
    for i in table.columns:
        if i in [f'avg_quantity_{time_1}'] or i in [f'avg_quantity_{time_2}']:
            table[i] = table[i].round(2).astype(str)
        elif table[i].dtypes == float:
            table[i] = (table[i] * 100).round(2).astype(str) + '%'

    return table


'''  說明2-6 繪製雙排柱樁圖，最多可以將6個績效指標繪圖，並於同一頁面呈現  '''

def draw_double_bar(pictures, column_label, column_bar1, column_bar2, y_label):
    num_pictures = min(len(pictures), 6)
    num_rows = (num_pictures + 1) // 2
    num_cols = min(num_pictures, 2)

    fig, axes = plt.subplots(num_rows, num_cols, figsize = (12, 6 * num_rows))
    if num_pictures == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    width = 0.3

    for i in range(num_pictures):
        j = pictures[i]
        x = np.arange(len(j.iloc[:, column_label]))
        x_label = j.iloc[:, column_label]
        x_bar_1 = j.iloc[:, column_bar1]
        x_bar_2 = j.iloc[:, column_bar2]
        x_name = j.columns[0] 

        axes[i].bar(x, x_bar_1, width, color = 'aquamarine', label = j.iloc[:, column_bar1].name)
        axes[i].bar(x + width, x_bar_2, width, color = 'dodgerblue', label = j.iloc[:, column_bar2].name)
        axes[i].set_xticks(x + width / 2)
        axes[i].set_xticklabels(x_label, rotation = 45, ha = 'right')
        axes[i].set_ylabel(y_label)
        axes[i].set_title(x_name)
        axes[i].legend(bbox_to_anchor = (1, 1), loc = 'upper left')

    plt.tight_layout()
    return plt.show()


'''  說明2-7 簡易業績分析圖表
此函數結合2-3報表, 2-4KPI清單, 2-5可讀性, 2-6視覺化圖四步驟，透過設定 index_cols, aggfunc_pick來簡化圖表產出語法
(1)index_cols
   最多設定2個維度，如果維度等於2或資料列數大於12，則產出時，會篩選出KPI review list再繪製圖表
(2)aggfunc_pick - 
   可從2-3的aggfunc中挑選想觀察的績效指標，最多設定6個　
'''

def draw_and_report(index_cols, aggfunc_pick):       
        
    if index_cols == ['month']:
        sales_review = sales_report(table, index_cols, cols, aggfunc_pick).sort_values(by = 'month')
    else:
        sales_review = sales_report(table, index_cols, cols, aggfunc_pick)
        
    pictures = []
    if len(index_cols) == 2:
        sales_review[f'{index_cols[0]}_{index_cols[1]}'] = (sales_review[index_cols[0]] + '_' + 
                                                            sales_review[index_cols[1]])
        if sales_review[f'{index_cols[0]}_{index_cols[1]}'].nunique() > 12:
            sales_review = sales_review.sort_values(by = 'revenue_diff', ascending = True).head(10)   
        elif sales_review[f'{index_cols[0]}_{index_cols[1]}'].nunique() <= 12:   
            sales_review = sales_review.sort_values(by = 'revenue_diff', ascending = True)
        for i in aggfunc_pick:
            pictures.append(sales_review.loc[:, [f'{index_cols[0]}_{index_cols[1]}', f'{i}_{time_2}', f'{i}_{time_1}']])     
              
    elif len(index_cols) == 1:
        if table[index_cols[0]].nunique() > 12:
            sales_review = sales_review.sort_values(by = 'revenue_diff', ascending = True).head(10)   
        elif table[index_cols[0]].nunique() <= 12:
            if index_cols == ['month']:
                sales_review = sales_review
            else:
                sales_review = sales_review.sort_values(by = 'revenue_diff', ascending = True)
        for i in aggfunc_pick:
            pictures.append(sales_review.loc[:, [index_cols[0], f'{i}_{time_2}', f'{i}_{time_1}']]) 
            
    elif len(index_cols) > 2:
        raise ValueError(f'index_cols參數上限為2個，目前為{len(index_cols)}個，請重新調整')
         
    draw_double_bar(pictures, 0, 1, 2, '')
    
    if len(index_cols) == 2:
        print(f'{index_cols[0]}_{index_cols[1]} 業績 {sales_review.pipe(data_types_transform)}')
    elif len(index_cols) == 1:
        print(f'{index_cols} 業績 {sales_review.pipe(data_types_transform)}')


########################################## 業績的分析 ##########################################

#變數設定
table = orders
index_cols = ['order_id']
cols = 'year'

time_1 = table[cols].max()
time_2 = table[cols].min()

#檢查變數與欄位
check_columns()
check_variables()

#產出包含所有績效指標的報表底稿
aggfunc_pick = ['revenue', 'revenue_share', 'profit', 'profit_rate', 'discount', 'discount_rate', 'avg_quantity', 'avg_sell_price']
sales_review = sales_report(table, index_cols, cols, aggfunc_pick)


'''  說明3-1-1  2023年業績 - 整體
(1)營收539萬美元，較2022年的528萬美元，yoy上升2.1%，遠低於該年度美國電商yoy的9.0%
(2)淨利率6.1%，較2022年的6.0%，yoy上升1.6%
(3)折扣率6.7%，較2022年的7.0%，yoy下降4.2%
(4)平均售價218美元，較2022年的210美元，yoy上升3.8%
(5)平均訂單量3.79，與2022年一致

該電商在售價小幅上升, 折扣小幅下降的情況下，獲利微幅上升，營收成長低於市場平均表現
'''

#初步檢視報表的數值
sta_1 = orders.query(f'year == {time_1}').select_dtypes('number').mean().T
sta_2 = orders.query(f'year == {time_2}').select_dtypes('number').mean().T

sales_review.hist(bins= 10, figsize = (12, 6))
plt.tight_layout()
plt.show()

columns_name = []
for i in ['revenue', 'profit', 'discount']:
    for j in [time_2, time_1]:
        columns_name.append(f'{i}_{j}')

sales_review = sales_review.fillna(0)
perfomance = sales_review.loc[:, columns_name].astype(int).sum()

print(f'業績概述 {perfomance}')  
print(f'淨利率_{time_1} {(perfomance.iloc[3]/perfomance.iloc[1]*100).round(1)}%')  
print(f'淨利率_{time_2} {(perfomance.iloc[2]/perfomance.iloc[0]*100).round(1)}%')  
print(f'折扣率_{time_1} {(perfomance.iloc[5]/perfomance.iloc[1]*100).round(1)}%')  
print(f'折扣率_{time_2} {(perfomance.iloc[4]/perfomance.iloc[0]*100).round(1)}%')  
print(f'平均售價_{time_1} {sta_1.iloc[7].round(1)}')  
print(f'平均售價_{time_2} {sta_2.iloc[7].round(1)}')  
print(f'平均訂單量_{time_1} {sta_1.iloc[2].round(2)}')  
print(f'平均訂單量_{time_2} {sta_2.iloc[2].round(2)}')  

    
'''  說明3-1-2  2023年業績 - 整體 - 各維度展開
(1)若從整體 x 各維度來看
   業績主力(營收最高且營收, 淨利率皆成長)為 region_East, category_Technology, ship_mode_Standard Class, segment_Consumer
   觀察對象(營收與淨利率衰退較多)為 category_Furniture, ship_mode-Second Class
   稍後將在說明3-2討論category_Furniture業績問題

(2)若從折扣率 x 各維度來看，可見前述該電商2023年折扣率的降低，是全面性政策，而非針對某一維度推行

(3)若從客群 x 各維度來看
   region     Consumer在多數地理區皆為主力客群
   category   Consumer以購買Furniture為大宗，Corporate, Home Office則為Technology業績成長的主因
   ship_mode  不論哪種客群，都偏好Standard Class的配送模式，且營收由配送速度慢到快遞減，由此現象推測該電商為分級收取運費制
'''

#整體 x 各維度檢視
for i in ['region', 'category', 'ship_mode', 'segment']:
    index_cols = [i]
    aggfunc_pick = ['revenue', 'revenue_share', 'profit_rate', 'avg_quantity', 'discount_rate', 'avg_sell_price']
    draw_and_report(index_cols, aggfunc_pick)

#segment x 其他維度 檢視
for i in ['region', 'category', 'ship_mode']:
    index_cols = ['segment', i]
    aggfunc_pick = ['revenue', 'revenue_share', 'profit_rate', 'avg_quantity', 'discount_rate', 'avg_sell_price']
    draw_and_report(index_cols, aggfunc_pick)


'''  說明3-1-3 2023年業績 - 整體 - month展開
若從月份來看
該電商絕大多數的月份都出現折扣率下降的現象
在一般月份並不特別影響業績，然而，在電商旺季時，相對於競爭對手的促銷，這樣的做法將使營收相對下降
凸顯該電商此策略的一大挑戰

#美國電商旺季大致為2(情人節), 8-9(返校季), 11(黑色星期五),12(聖誕節)月
'''

index_cols = ['month']
aggfunc_pick = ['revenue', 'revenue_share', 'profit_rate', 'avg_quantity', 'discount_rate', 'avg_sell_price']
draw_and_report(index_cols, aggfunc_pick)

    
'''  說明3-2-1 2023年category_Furniture業績問題 - sub_category展開 
若從Furniture的子類別來看
(1)Chairs, Bookcases在淨利率下降的情況下，營收yoy分別成長7.6%, 15.4%，符合美國電商營收yoy成長水準
(2)Tables堅守獲利，故營收下降18萬，yoy下降29.2%，
(3)Furnishings則營收, 淨利率, 訂單商品量, 價格都較2022年下降，值得注意

可知Furniture的業績問題來自Tables, Furnishings兩個子類別
'''

table = orders.copy().query('category == "Furniture"')

index_cols = ['sub_category']
aggfunc_pick = ['revenue', 'revenue_share', 'profit_rate', 'avg_quantity', 'discount_rate', 'avg_sell_price']
draw_and_report(index_cols, aggfunc_pick)


'''  說明3-2-2 2023年category_Furniture_Tables, Furnishings業績問題 - 各維度展開 
若從Tables, Furnishings x 各維度來看
region    West, South是營收衰退最明顯的區域
ship_mode 各配送方式都衰退，以Standard Class營收衰退最嚴重
segment   幾乎是全面性的營收衰退，連讓利促銷都無法挽回此情形
綜合以上三個維度的觀察，此衰退與客群差異較無關

於是進一步觀察region x ship_mode，並以整體訂單與Furniture_Tables, Furnishings來比較
可明顯看出，在兩者的淨利率都大約維持平盤的情況下，
整體訂單的West, South x ship_mode組合，業績約衰退21-38%；而Furniture_Tables, Furnishings，則多數衰退57-83%
故推測Furniture_Tables, Furnishings與region x ship_mode有關
'''

#Tables, Furnishings x 其他維度檢視
table = orders.copy().query('category == "Furniture" and (sub_category == "Tables" or sub_category == "Furnishings")')

for i in ['region', 'ship_mode', 'segment']:
    index_cols = ['sub_category', i]
    aggfunc_pick = ['revenue', 'revenue_share', 'profit_rate', 'avg_quantity', 'discount_rate', 'avg_sell_price']
    draw_and_report(index_cols, aggfunc_pick)

#整體_region x ship_mode檢視
table = orders

index_cols = ['region', 'ship_mode']
aggfunc_pick = ['revenue', 'revenue_share', 'profit_rate', 'avg_quantity', 'discount_rate', 'avg_sell_price']
draw_and_report(index_cols, aggfunc_pick)

#Tables, Furnishings_region x ship_mode檢視
table = orders.copy().query('category == "Furniture" and (sub_category == "Tables" or sub_category == "Furnishings")')

index_cols = ['region', 'ship_mode']
aggfunc_pick = ['revenue', 'revenue_share', 'profit_rate', 'avg_quantity', 'discount_rate', 'avg_sell_price']
draw_and_report(index_cols, aggfunc_pick)


'''  說明3-2-3 2023年category_Furniture_Tables, Furnishings業績問題 - 小結
region x ship_mode的交集點就是運費
Furniture_Tables, Furnishings這樣的大材積商品，更容易因運費而影響購買意願

在美國，大型電商(如Amazon)會用會員制度或免運門檻來降低此障礙
中小型電商則往往需要購買者自行負擔

Furniture_Tables, Furnishings的業績問題
推測是該電商調整商品的運費制度，導致West, South區域的大材積商品運費明顯上漲，讓原本為業績主力的Furniture業績受挫

面對此問題
短期來看，如果能說服供應商一同補貼運費，來刺激消費者購物意願，或許能拉抬業績
長期來說，嘗試以RFM模型來劃分消費者，規劃對應的會員制度來降低運費的衝擊，或許是更好的方法
''' 
