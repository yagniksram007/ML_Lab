# Demonstrate data preprocessing (data cleaning, integration,and transformation) operations on a suitable data.

import pandas as pd

customers_data = {
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Yagnik', 'Nikith', 'Abyan', 'Yujith', 'Naizil'],
    'age': [30, 25, 40, None, 35],
    'email': ['yagnik@gmail.com', 'Nikki@gmail.com', 'abyan@gmail.com', 'yujith@gmail.com', None]
}

orders_data = {
    'order_id': [101, 102, 103, 104, 105],
    'customer_id': [1, 2, 3, 4, 5],
    'product_id': [201, 202, 203, 204, 205],
    'quantity': [2, 1, 3, 2, 1]
}

products_data = {
    'product_id': [201, 202, 203, 204, 205],
    'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Printer'],
    'price': [1000, 20, 50, 300, 200]
}

customers_df = pd.DataFrame(customers_data)
orders_df = pd.DataFrame(orders_data)
products_df = pd.DataFrame(products_data)

customers_df['age'].fillna(customers_df['age'].mean(), inplace=True)
customers_df['email'].fillna('N/A', inplace=True)

merged_df = pd.merge(pd.merge(customers_df, orders_df, on='customer_id'), products_df, on='product_id')


merged_df['total_price'] = merged_df['quantity'] * merged_df['price']

print("Cleaned, Integrated and Transformed Data:")
print(merged_df)


# Output:
# Cleaned, Integrated and Transformed Data:
#    customer_id    name   age  ... product_name  price  total_price
# 0            1  Yagnik  30.0  ...       Laptop   1000         2000
# 1            2  Nikith  25.0  ...        Mouse     20           20
# 2            3   Abyan  40.0  ...     Keyboard     50          150
# 3            4  Yujith  32.5  ...      Monitor    300          600
# 4            5  Naizil  35.0  ...      Printer    200          200

# [5 rows x 10 columns]