
########################################## Data Preprocessing ##########################################

'''  Description 1-1 Retrieve and read the U.S. e-commerce sales order data from Kaggle.  '''

#import the packages
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
        
pd.set_option('display.float_format', '{:.2f}'.format)

#read the data
path = kagglehub.dataset_download("ankitbansal06/retail-orders").replace('\\', '/')

data = []
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('csv'):
            data.append(os.path.join(dirname, filename))
        
orders = pd.read_csv(data[0])


'''  Description 1-2 Obtain basic information about the data.  '''

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


'''  Description 1-3 Perform data processing to improve data quality.  '''

#standardize the data column name formats to improve readability
orders.columns = [i.replace(' ', '_').lower() for i in orders.columns]
print(orders.columns)

#ensure that the data column attributes are correct
orders['order_date'] = pd.to_datetime(orders['order_date'], errors='coerce')
orders['order_id'] = orders['order_id'].astype(str)
orders['postal_code'] = orders['postal_code'].astype(str)
print(orders.dtypes)

#remove the 5 rows with outliers
orders = orders[~orders['ship_mode'].isin(['Not Available', 'unknown'])]
print(orders['ship_mode'].value_counts())

#fill the missing value with the mode
print(orders.isna().sum().sort_values(ascending = False).head(10))
orders['ship_mode'] = orders['ship_mode'].fillna(orders['ship_mode'].mode()[0])
print(orders[orders['ship_mode'].isna() == True])


'''  Description 1-4  
(1)To ensure the quality of subsequent analysis, set and define the columns, and perform initial 
   calculations on the data.
(2)The average gross profit margin for the Furniture, Office Supplies, and Technology categories 
   in U.S. e-commerce in 2023 is approximately 32%, with a net profit margin of about 6.0%.
   Therefore, the discount rate in the report is set to 6.9%, and the profit_rate is defined as 
   the net profit margin, with the net profit margin for the order data set to 6.0%.
'''

#add columns and setting the discount rate column
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

print(f'Monthly Revenue {orders.groupby("month")["revenue"].sum().astype(int)}')
print(f'Net profit margin {(orders["profit"].sum() / orders["revenue"].sum() * 100).round(2)}%') 
print(f'Discount rate {(orders["discount"].sum() / orders["revenue"].sum() * 100).round(2)}%') 
print(f'Net profit Margin (Positive/Negative) Ratio {(orders["profit"] > 0).value_counts(normalize = True).round(2) * 100}') 


########################################## Function Development ##########################################


'''  Description 2-1 A simple fail-safe check to ensure that all columns in the raw data exist. ''' 

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
            print(f'{i} column does not exist; please check the raw data')     
        
    for j in columns_check_number:
        if table[j].sum() > 0:
            print(f'{j}  sum {table[j].sum().astype(int)}')
        else:
            print(f'{j} column does not exist; please check the raw data')
            
            
'''  Description 2-2 A simple fail-safe check to ensure that all variables have been declared.  '''

def check_variables():
        
    for i in ['table', 'index_cols', 'cols']:
        try:
            eval(i)
        except NameError:
            print(f'{i} column does not exist; please declare the variable')
            exit()
    print('The variable has been defined; please execute the report template')


'''  Description 2-3 Create a report template that presents indicators using MoM, YoY, and difference methods  '''

def sales_report(table, index_cols, cols, aggfunc_pick):
        
    #pivot Table
    aggfunc = {'revenue' : ['sum'], 
               'profit' : ['sum'],
               'cost' : ['sum'],
               'discount' : ['sum'],
               'quantity' : ['mean'],
               'list_price' : ['mean'],
               'cost_price' : ['mean'],
               'sell_price' : ['mean']}

    table = table.pivot_table(index = index_cols, columns = cols, aggfunc = aggfunc)
    
    #define the column names for subsequent processing
    table_columns_percentage = list(set([(f'{i}_{j}').replace('_sum', "")
                                                     .replace('mean', 'avg') 
                                    for i, j, k in table.columns.to_flat_index()]))
    table_columns_diff = ['revenue', 'profit_rate']

    #flatten the column names for subsequent processing                                                             
    table.columns = ([(f'{i}_{j}_{k}').replace('_sum', "").replace('mean', 'avg') 
                                       for i, j, k in table.columns.to_flat_index()])
   
    #add performance-related columns
    for i in [time_1, time_2]:
        #Revenue share
        table[f'revenue_share_{i}'] = np.where(table[f'revenue_{i}'] != 0, 
                                      (table[f'revenue_{i}'] / table[f'revenue_{i}'].sum()), 0) 
        #Net profit margin
        table[f'profit_rate_{i}']   = np.where(table[f'revenue_{i}'] != 0, 
                                      (table[f'profit_{i}'] / table[f'revenue_{i}']), 0)
        #Discount rate
        table[f'discount_rate_{i}'] = np.where(table[f'revenue_{i}'] != 0, 
                                      (table[f'discount_{i}'] / table[f'revenue_{i}']), 0)
                         
    #add YoY and MoM columns
    for i in table_columns_percentage + ['revenue_share', 'profit_rate', 'discount_rate']:
        if cols == 'year':                                                                                                
            table[f'{i}_yoy'] = np.where(table[f'{i}_{time_2}'] != 0, 
                                (table[f'{i}_{time_1}'] - table[f'{i}_{time_2}']) / table[f'{i}_{time_2}'], 0) 
        elif cols == 'month':                                                                                                
            table[f'{i}_mom'] = np.where(table[f'{i}_{time_2}'] != 0, 
                                (table[f'{i}_{time_1}'] - table[f'{i}_{time_2}']) / table[f'{i}_{time_2}'], 0)     
                                                                                                      
    #add diff(delta) columns                                                                                                         
    for i in table_columns_diff:
        table[f'{i}_diff'] = table[f'{i}_{time_1}'] - table[f'{i}_{time_2}']
    
    #adjust the column names to a common format to enhance readability  
    for i in table.columns:
        if 'avg' in i:
            table.rename(columns = {i : f'avg_{i.replace("_avg", "")}'}, inplace = True)
            
    #setting the report columns to be used and set the sorting order
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
                  
    #generate the report
    table = table.loc[:, table_columns_name].reset_index().sort_values(by = sort, ascending = False)
    
    return table


'''  Description 2-4 
Set KPI conditions and generate Star/Review lists to identify the strengths and weaknesses of sales performance
(1)Star list KPI conditions:
 1.YoY, MoM revenue and net profit margin both increase by more than 10%.
 2.YoY, MoM revenue increase by more than 20%.
                        
(2)Review list KPI conditions:
 1.YoY, MoM revenue and net profit margin both decrease by more than 10%.
 2.YoY, MoM revenue decrease by more than 20%.
 3.YoY, MoM net profit rate decrease by more than 20%.
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


'''  Description 2-5 Enhance the readability of the report by setting columns as integers or as floats.  '''

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


'''  Description 2-6 Up to 6 indicators can be specified and each will produce an individual dual-bar chart.  '''

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


'''  Description 2-7
This function combines steps 2-3 (report generation), 2-4 (KPI list), 2-5 (readability), and 2-6 (visualization) 
by simplifying the chart and report generation syntax through the use of index_cols and aggfunc_pick settings
(1)index_cols:
   Set up to 2 dimensions. If the number of dimensions equals 2 or the number of rows exceeds 12, 
   the function will use the KPI review list sorting by revenue_diff to generate the chart.
(2)aggfunc_pick
   Select the indicators to observe from the aggfunc in step 2-3, with a maximum of 6 indicators
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
        raise ValueError(f'The maximum limit for the index_cols parameter is 2, there are {len(index_cols)}. Please adjust accordingly')

    draw_double_bar(pictures, 0, 1, 2, '')
    
    if len(index_cols) == 2:
        print(f'{index_cols[0]}_{index_cols[1]} sales {sales_review.pipe(data_types_transform)}')
    elif len(index_cols) == 1:
        print(f'{index_cols} sales {sales_review.pipe(data_types_transform)}')


########################################## Sales data analysis ##########################################

#variable declaration
table = orders
index_cols = ['order_id']
cols = 'year'

time_1 = table[cols].max()
time_2 = table[cols].min()

#fail-safe check
check_columns()
check_variables()

#generate the report
aggfunc_pick = ['revenue', 'revenue_share', 'profit', 'profit_rate', 'discount', 'discount_rate', 'avg_quantity', 'avg_sell_price']
sales_review = sales_report(table, index_cols, cols, aggfunc_pick)


'''  Description 3-1-1  Sales performance in 2023 - overall
(1)Revenue was 5.39 million, up 2.1% YoY from 5.28 million in 2022, 
   significantly lower than the 9.0% YoY growth of the U.S. e-commerce market in the same year.
(2)Net profit margin was 6.1%, a 1.6% YoY increase compared to 6.0% in 2022.
(3)Discount rate was 6.7%, a 4.2% YoY decrease compared to 7.0% in 2022.
(4)Average selling price was 218, a 3.8% YoY increase from 210 in 2022.
(5)Average order quantity was 3.79, consistent with 2022.

The e-commerce company's slightly higher prices and reduced discounts resulted in a small profit gain, 
though its revenue growth was below the market average.
'''

#preliminary review of the report figures
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

print(f'sales overview {perfomance}')  
print(f'Net profit rate_{time_1} {(perfomance.iloc[3]/perfomance.iloc[1]*100).round(1)}%')  
print(f'Net profit rate_{time_2} {(perfomance.iloc[2]/perfomance.iloc[0]*100).round(1)}%')  
print(f'discount rate_{time_1} {(perfomance.iloc[5]/perfomance.iloc[1]*100).round(1)}%')  
print(f'discount_rate_{time_2} {(perfomance.iloc[4]/perfomance.iloc[0]*100).round(1)}%')  
print(f'average price_{time_1} {sta_1.iloc[7].round(1)}')  
print(f'average peice_{time_2} {sta_2.iloc[7].round(1)}')  
print(f'average items per order_{time_1} {sta_1.iloc[2].round(2)}')  
print(f'average items per order_{time_2} {sta_2.iloc[2].round(2)}')  

    
'''  Description 3-1-2  Sales performance in 2023 - overall x dimensions
(1)Overall x dimensional perspective
   The main performance drivers are region_East, category_Technology, ship_mode_Standard Class, and segment_Consumer.
   The areas of concern are category_Furniture and ship_mode_Second Class.
   The performance issues of category_Furniture will be discussed later in Description 3-2.

(2)Discount rate x dimensions perspective
   The decrease in the e-commerce company's discount rate in 2023 was a company-wide policy 
   rather than a strategy targeted at a specific dimension

(3)Segment x dimensions perspective
   region:    Consumer is the main customer segment across most geographic regions.
   category:  Consumers primarily purchase Furniture, 
              while Corporate and Home Office customers drive revenue growth mainly in the Technology category.
   ship_mode: Regardless of the customer segment, there is a preference for the Standard Class shipping mode. 
              Additionally, revenue decreases as shipping speed increases, suggesting that the e-commerce platform 
              likely adopts a tiered shipping fee structure.
'''

#overall x dimensions
for i in ['region', 'category', 'ship_mode', 'segment']:
    index_cols = [i]
    aggfunc_pick = ['revenue', 'revenue_share', 'profit_rate', 'avg_quantity', 'discount_rate', 'avg_sell_price']
    draw_and_report(index_cols, aggfunc_pick)

#segment x dimensions
for i in ['region', 'category', 'ship_mode']:
    index_cols = ['segment', i]
    aggfunc_pick = ['revenue', 'revenue_share', 'profit_rate', 'avg_quantity', 'discount_rate', 'avg_sell_price']
    draw_and_report(index_cols, aggfunc_pick)


'''  Description 3-1-3 Sales performance in 2023 - overall x month
Overall x month perspective
The e-commerce platform's discount rates generally decrease each month. 
While this does not significantly impact revenue during regular months, 
during peak shopping seasons, this strategy could lead to revenue decline when competitors' promotions. 
This highlights a key challenge for the platform’s strategy in 2023.

#The peak shopping seasons for U.S. e-commerce generally occur in 
 Feb(Valentine's Day), Aug to Sep(Back-to-School season), Nov(Black Friday), and Dec(Christmas)
'''

index_cols = ['month']
aggfunc_pick = ['revenue', 'revenue_share', 'profit_rate', 'avg_quantity', 'discount_rate', 'avg_sell_price']
draw_and_report(index_cols, aggfunc_pick)

    
'''  Description 3-2-1 sales performance in 2023 - category_Furniture x sub_category 
In the Furniture category, 
(1)both Chairs and Bookcases experienced a decline in net profit margin, but their YoY revenue grew by 7.6% and 15.4% respectively,
   which aligns with the overall YoY revenue growth levels in U.S. e-commerce.
(2)Tables, maintained profitability, but revenue decreased by 18k, resulting in a YoY decline of 29.2%.
(3)Furnishings saw declines across multiple metrics compared to 2022, 
   including revenue, net profit margin, order quantity, and pricing, which needs attention.

It is clear that the category_Furniture issues are mainly driven by the Tables and Furnishings subcategories.
'''

table = orders.copy().query('category == "Furniture"')

index_cols = ['sub_category']
aggfunc_pick = ['revenue', 'revenue_share', 'profit_rate', 'avg_quantity', 'discount_rate', 'avg_sell_price']
draw_and_report(index_cols, aggfunc_pick)


'''  Description 3-2-2 sales performance in 2023 - category_Furniture x dimensions
Region:        The West and South regions have the biggest revenue declines.
Shipping Mode: All shipping methods declined, with Standard Class seeing the largest drop.
Segment:       Revenue declined across most segments, with even discounts unable to reverse the trend.
Based on this analysis, the decline seems unrelated to customer segment differences.

When comparing overall orders with Furniture_Tables and Furnishings, 
the West and South regions showed a revenue decline of 21-38% across all shipping modes. 
For Furniture_Tables and Furnishings, the decline was much steeper, ranging from 57-83%. 
This suggests that the performance issues in these two subcategories are related to the region x ship mode combination.
'''

#Tables, Furnishings x dimensions
table = orders.copy().query('category == "Furniture" and (sub_category == "Tables" or sub_category == "Furnishings")')

for i in ['region', 'ship_mode', 'segment']:
    index_cols = ['sub_category', i]
    aggfunc_pick = ['revenue', 'revenue_share', 'profit_rate', 'avg_quantity', 'discount_rate', 'avg_sell_price']
    draw_and_report(index_cols, aggfunc_pick)

#Overall x region_ship_mode
table = orders

index_cols = ['region', 'ship_mode']
aggfunc_pick = ['revenue', 'revenue_share', 'profit_rate', 'avg_quantity', 'discount_rate', 'avg_sell_price']
draw_and_report(index_cols, aggfunc_pick)

#Tables, Furnishings_region x ship_mode
table = orders.copy().query('category == "Furniture" and (sub_category == "Tables" or sub_category == "Furnishings")')

index_cols = ['region', 'ship_mode']
aggfunc_pick = ['revenue', 'revenue_share', 'profit_rate', 'avg_quantity', 'discount_rate', 'avg_sell_price']
draw_and_report(index_cols, aggfunc_pick)


'''  Description 3-2-3 sales performance in 2023 - category_Furniture - conclusion
The intersection of region x ship mode represents shipping costs.
For large items like Tables and Furnishings, shipping costs are more likely to impact customers' willingness to purchase.

In the U.S., large e-commerce platforms (e.g., Amazon) reduce this barrier by offering membership programs or free shipping thresholds.
Smaller e-commerce platforms, often require buyers to cover the shipping costs themselves.

The Tables and Furnishings issues are likely due to the e-commerce platform's adjustment of the shipping fee policy. 
This led to a significant increase in shipping costs for large items in the West and South regions, which affected
the sales of Furniture, a previously strong-performing category.

To address this issue:
In the short term, convincing suppliers to jointly subsidize shipping costs could help stimulate consumer purchasing willingness.
In the long term, using an RFM model to segment consumers and planning a corresponding membership program to reduce 
the impact of shipping costs might be a better solution.
''' 
