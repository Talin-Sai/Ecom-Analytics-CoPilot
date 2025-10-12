-- models/marts/monthly_sales.sql
-- Purpose: Monthly sales (sum of item price), using order timestamps
{{ config(materialized='view') }}
with items as (
    select
        oi.order_id,
        oi.price::numeric
    from public.olist_order_items_dataset oi
),

orders as (
    select
        o.order_id,
        -- Some Olist dumps store this as text; cast safely to timestamp
        date_trunc('month', (o.order_purchase_timestamp)::timestamp) as month
    from public.olist_orders_dataset o
)

select
    o.month::date as month,
    sum(i.price)::numeric(12,2) as sales
from items i
join orders o using (order_id)
group by 1
order by 1

