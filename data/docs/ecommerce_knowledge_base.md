# E-Commerce Business Knowledge Base

## Key Metrics & Definitions

### Order Fulfillment Rate
The order fulfillment rate is the percentage of orders that were successfully delivered to the customer.
Formula: `fulfilled_orders / total_orders × 100`
A healthy fulfillment rate is above 85%. Rates below 70% indicate supply chain or logistics issues.

### Average Order Value (AOV)
AOV is the average revenue earned per order.
Formula: `SUM(total_amount) / COUNT(orders)` for non-cancelled, non-refunded orders.
AOV can be improved through upselling, cross-selling, and bundling strategies.

### Customer Lifetime Value (LTV / CLV)
Customer lifetime value is the total revenue a business can expect from a single customer account.
Formula: `SUM(orders.total_amount)` where status = 'delivered', grouped by customer.
Premium customers typically have 2–3× higher LTV than standard customers.

### Gross Margin
Gross margin measures profitability per product.
Formula: `(price - cost) / price × 100`
A margin above 40% is generally healthy for e-commerce.

### Return/Refund Rate
The percentage of orders that were refunded.
Formula: `refunded_orders / total_orders × 100`
High refund rates (>10%) often indicate product quality issues or misleading listings.

### Cart Abandonment
Not tracked in this database, but cart abandonment typically occurs when customers add items but don't complete checkout.
Industry average abandonment rate: ~70%.

---

## Product Categories

- **Electronics**: Highest average order value (~$120), moderate return rate
- **Clothing**: Highest return rate (~15%), size/fit issues common
- **Home & Garden**: Seasonal demand spikes in spring/summer
- **Sports**: Strong January demand (New Year resolutions)
- **Books**: Lowest return rate (<2%), highest margin product category
- **Beauty**: Subscription-friendly, strong repeat purchase behaviour
- **Toys**: Q4 seasonal spike (Oct–Dec), up to 40% of annual sales
- **Food & Grocery**: Lowest AOV, highest purchase frequency

---

## Customer Segments

### Premium Customers
- Defined by `customers.is_premium = true`
- Represent ~15% of the customer base
- Receive automatic 10% discount on all orders
- Typically have 2–4× higher purchase frequency
- Should be prioritised for loyalty programmes

### Standard Customers
- Represent ~85% of the customer base
- Can be upgraded to premium via marketing campaigns

---

## Order Statuses

| Status    | Description                                         |
|-----------|-----------------------------------------------------|
| pending   | Order placed, payment not yet confirmed             |
| confirmed | Payment received, awaiting fulfilment               |
| shipped   | Order dispatched, in transit                        |
| delivered | Order received by customer                          |
| cancelled | Order cancelled before shipment                     |
| refunded  | Order returned and money refunded                   |

---

## Shipping & Countries

The platform ships to: US, UK, DE, FR, CA, AU, NL, SE, JP, SG.
Shipping time varies by country:
- Domestic (US, UK): 2–5 business days
- Europe (DE, FR, NL, SE): 3–7 business days
- Asia-Pacific (JP, SG, AU): 7–14 business days

---

## Frequently Asked Questions

### How do I find the best-selling products?
Query the `order_items` table joined with `products` and `orders`, summing `line_total` grouped by product name. Exclude cancelled orders.

### How do I calculate revenue for a specific period?
Use `WHERE orders.created_at BETWEEN 'start_date' AND 'end_date'` and sum `total_amount` or `line_total`.

### What is a good rating threshold for product quality?
Products with an average rating below 3.0 should be reviewed. Products above 4.5 are top performers.

### How are discounts applied?
Premium customers receive a 10% discount automatically, stored in `orders.discount_amount`.
The `total_amount` field already reflects the post-discount price.

### What does `line_total` represent?
`line_total = quantity × unit_price` for a single order item. Sum all `line_total` values in an order to get pre-discount gross revenue.

---

## Data Dictionary

| Table        | Key Columns                                                              |
|--------------|--------------------------------------------------------------------------|
| customers    | id, email, first_name, last_name, country, city, is_premium, created_at |
| products     | id, name, sku, category_id, price, cost, stock, is_active                |
| categories   | id, name, description                                                    |
| orders       | id, customer_id, status, total_amount, discount_amount, shipping_country |
| order_items  | id, order_id, product_id, quantity, unit_price, line_total               |
| reviews      | id, customer_id, product_id, rating (1–5), comment, created_at          |
