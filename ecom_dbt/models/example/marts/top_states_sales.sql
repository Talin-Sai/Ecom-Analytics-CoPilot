{{ config(materialized='view') }}

with
items as (
    select
        oi.order_id,
        oi.price::numeric as price
    from public.olist_order_items_dataset oi
),
orders as (
    select
        o.order_id,
        o.customer_id
    from public.olist_orders_dataset o
),
customers as (
    select
        c.customer_id,
        c.customer_state
    from public.olist_customers_dataset c
)

select
    c.customer_state,
    sum(i.price)::numeric(12,2) as sales
from items i
join orders o using (order_id)
join customers c using (customer_id)
group by 1
order by sales desc
