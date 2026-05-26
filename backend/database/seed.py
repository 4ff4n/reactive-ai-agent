"""
Seed the database with realistic synthetic e-commerce data.
Run: python -m backend.database.seed
"""
import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal

from faker import Faker
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import AsyncSessionLocal, init_db
from backend.database.models import (
    Category, Customer, Order, OrderItem, OrderStatus, Product, Review,
)

fake = Faker()
random.seed(42)
Faker.seed(42)

# ── catalogue ────────────────────────────────────────────────────────────────

CATEGORIES = [
    ("Electronics",     "Gadgets, devices and accessories"),
    ("Clothing",        "Apparel for men, women and kids"),
    ("Home & Garden",   "Furniture, decor and garden tools"),
    ("Sports",          "Fitness and outdoor equipment"),
    ("Books",           "Fiction, non-fiction and textbooks"),
    ("Beauty",          "Skincare, haircare and cosmetics"),
    ("Toys",            "Toys and games for all ages"),
    ("Food & Grocery",  "Packaged food and pantry staples"),
]

PRODUCT_TEMPLATES = {
    "Electronics": [
        ("Wireless Earbuds Pro", 89.99, 32.00),
        ("USB-C Hub 7-in-1",     49.99, 18.00),
        ("Smart Watch Series X", 199.99, 72.00),
        ("Portable Charger 20k", 39.99, 14.00),
        ("Mechanical Keyboard",  129.99, 47.00),
        ("Webcam HD 1080p",      69.99, 25.00),
    ],
    "Clothing": [
        ("Classic Cotton Tee",   24.99,  7.00),
        ("Slim Fit Chinos",      59.99, 18.00),
        ("Hooded Fleece Jacket", 79.99, 27.00),
        ("Running Shorts",       34.99, 10.00),
        ("Linen Summer Dress",   54.99, 17.00),
    ],
    "Home & Garden": [
        ("Bamboo Cutting Board",  29.99,  9.00),
        ("Cast Iron Skillet",     44.99, 16.00),
        ("LED Desk Lamp",         35.99, 12.00),
        ("Indoor Plant Pot Set",  22.99,  7.50),
        ("Memory Foam Pillow",    49.99, 17.00),
    ],
    "Sports": [
        ("Yoga Mat Pro",          39.99, 12.00),
        ("Resistance Band Set",   24.99,  7.00),
        ("Foam Roller",           19.99,  6.00),
        ("Water Bottle 1L",       22.99,  6.50),
        ("Jump Rope",             14.99,  4.00),
    ],
    "Books": [
        ("Clean Code",            35.99, 12.00),
        ("Atomic Habits",         18.99,  6.50),
        ("Dune",                  14.99,  5.00),
        ("Python Crash Course",   39.99, 14.00),
        ("Sapiens",               17.99,  6.00),
    ],
    "Beauty": [
        ("Vitamin C Serum",       32.99, 10.00),
        ("Hyaluronic Moisturiser",28.99,  9.00),
        ("Shampoo & Conditioner", 19.99,  6.00),
        ("Lip Balm Set",           9.99,  2.50),
        ("Face Mask Pack",        14.99,  4.00),
    ],
    "Toys": [
        ("LEGO City Set",         49.99, 18.00),
        ("Remote Control Car",    34.99, 11.00),
        ("Wooden Puzzle 100pc",   22.99,  7.00),
        ("Slime Kit",             16.99,  5.00),
        ("Board Game Classic",    29.99,  9.00),
    ],
    "Food & Grocery": [
        ("Organic Coffee Beans",  18.99,  7.00),
        ("Premium Olive Oil",     14.99,  5.50),
        ("Mixed Nuts 500g",       12.99,  4.50),
        ("Dark Chocolate Pack",    9.99,  3.00),
        ("Green Tea 50 bags",      8.99,  2.75),
    ],
}

COUNTRIES = ["US", "UK", "DE", "FR", "CA", "AU", "NL", "SE", "JP", "SG"]
STATUS_WEIGHTS = [
    (OrderStatus.delivered, 0.50),
    (OrderStatus.shipped,   0.15),
    (OrderStatus.confirmed, 0.10),
    (OrderStatus.pending,   0.10),
    (OrderStatus.cancelled, 0.10),
    (OrderStatus.refunded,  0.05),
]


def weighted_status() -> OrderStatus:
    statuses, weights = zip(*STATUS_WEIGHTS)
    return random.choices(statuses, weights=weights, k=1)[0]


def random_date(days_back: int = 365) -> datetime:
    return datetime.utcnow() - timedelta(days=random.randint(0, days_back))


# ── main seeder ──────────────────────────────────────────────────────────────

async def seed(session: AsyncSession) -> None:
    print("🌱  Starting seed…")

    # ── categories ──────────────────────────────────────────────────────────
    cat_map: dict[str, Category] = {}
    for name, desc in CATEGORIES:
        cat = Category(name=name, description=desc)
        session.add(cat)
        cat_map[name] = cat
    await session.flush()
    print(f"   ✓ {len(CATEGORIES)} categories")

    # ── products ────────────────────────────────────────────────────────────
    product_objs: list[Product] = []
    sku_counter = 1
    for cat_name, templates in PRODUCT_TEMPLATES.items():
        for prod_name, price, cost in templates:
            p = Product(
                name=prod_name,
                sku=f"SKU-{sku_counter:05d}",
                category_id=cat_map[cat_name].id,
                price=Decimal(str(price)),
                cost=Decimal(str(cost)),
                stock=random.randint(0, 500),
                is_active=random.random() > 0.05,
                description=fake.sentence(nb_words=12),
            )
            session.add(p)
            product_objs.append(p)
            sku_counter += 1
    await session.flush()
    print(f"   ✓ {len(product_objs)} products")

    # ── customers ───────────────────────────────────────────────────────────
    customers: list[Customer] = []
    for _ in range(500):
        c = Customer(
            email=fake.unique.email(),
            first_name=fake.first_name(),
            last_name=fake.last_name(),
            country=random.choice(COUNTRIES),
            city=fake.city(),
            is_premium=random.random() < 0.15,
            created_at=random_date(730),
        )
        session.add(c)
        customers.append(c)
    await session.flush()
    print(f"   ✓ {len(customers)} customers")

    # ── orders + order_items ────────────────────────────────────────────────
    orders_created = 0
    for customer in customers:
        n_orders = random.choices([0, 1, 2, 3, 4, 5], weights=[5, 30, 30, 20, 10, 5])[0]
        for _ in range(n_orders):
            order_date = random_date(365)
            n_items = random.randint(1, 5)
            chosen_products = random.sample(product_objs, min(n_items, len(product_objs)))
            items: list[OrderItem] = []
            total = Decimal("0")
            for prod in chosen_products:
                qty = random.randint(1, 3)
                line = Decimal(str(prod.price)) * qty
                items.append(OrderItem(
                    product_id=prod.id,
                    quantity=qty,
                    unit_price=prod.price,
                    line_total=line,
                ))
                total += line
            discount = total * Decimal("0.10") if customer.is_premium else Decimal("0")
            order = Order(
                customer_id=customer.id,
                status=weighted_status(),
                total_amount=total - discount,
                discount_amount=discount,
                shipping_country=customer.country,
                created_at=order_date,
                updated_at=order_date + timedelta(days=random.randint(0, 5)),
            )
            session.add(order)
            await session.flush()
            for item in items:
                item.order_id = order.id
                session.add(item)
            orders_created += 1
    await session.flush()
    print(f"   ✓ ~{orders_created} orders created")

    # ── reviews ─────────────────────────────────────────────────────────────
    reviews_created = 0
    sample_customers = random.sample(customers, 200)
    for customer in sample_customers:
        n_reviews = random.randint(0, 3)
        reviewed = random.sample(product_objs, min(n_reviews, len(product_objs)))
        for prod in reviewed:
            r = Review(
                customer_id=customer.id,
                product_id=prod.id,
                rating=random.choices([1, 2, 3, 4, 5], weights=[2, 5, 10, 35, 48])[0],
                comment=fake.sentence(nb_words=random.randint(8, 20)) if random.random() > 0.3 else None,
                created_at=random_date(180),
            )
            session.add(r)
            reviews_created += 1
    await session.flush()
    print(f"   ✓ {reviews_created} reviews")

    await session.commit()
    print("✅  Seed complete!")


async def main():
    await init_db()
    async with AsyncSessionLocal() as session:
        # Clear existing data
        await session.execute(text("TRUNCATE reviews, order_items, orders, products, customers, categories RESTART IDENTITY CASCADE"))
        await session.commit()
        await seed(session)


if __name__ == "__main__":
    asyncio.run(main())
