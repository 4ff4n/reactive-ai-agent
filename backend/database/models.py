"""
E-commerce schema: customers, products, categories, orders, order_items, reviews.
All tables include created_at/updated_at for temporal queries.
"""
from datetime import datetime
from decimal import Decimal
from sqlalchemy import (
    String, Integer, Numeric, Boolean, Text, DateTime,
    ForeignKey, func, Enum as SAEnum
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.database.connection import Base
import enum


class OrderStatus(str, enum.Enum):
    pending   = "pending"
    confirmed = "confirmed"
    shipped   = "shipped"
    delivered = "delivered"
    cancelled = "cancelled"
    refunded  = "refunded"


class Customer(Base):
    __tablename__ = "customers"

    id:           Mapped[int]      = mapped_column(Integer, primary_key=True, index=True)
    email:        Mapped[str]      = mapped_column(String(255), unique=True, nullable=False, index=True)
    first_name:   Mapped[str]      = mapped_column(String(100), nullable=False)
    last_name:    Mapped[str]      = mapped_column(String(100), nullable=False)
    country:      Mapped[str]      = mapped_column(String(100), nullable=False)
    city:         Mapped[str]      = mapped_column(String(100), nullable=False)
    is_premium:   Mapped[bool]     = mapped_column(Boolean, default=False)
    created_at:   Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at:   Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    orders:   Mapped[list["Order"]]  = relationship("Order", back_populates="customer")
    reviews:  Mapped[list["Review"]] = relationship("Review", back_populates="customer")


class Category(Base):
    __tablename__ = "categories"

    id:          Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name:        Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)

    products: Mapped[list["Product"]] = relationship("Product", back_populates="category")


class Product(Base):
    __tablename__ = "products"

    id:           Mapped[int]     = mapped_column(Integer, primary_key=True, index=True)
    name:         Mapped[str]     = mapped_column(String(255), nullable=False, index=True)
    sku:          Mapped[str]     = mapped_column(String(50), unique=True, nullable=False)
    category_id:  Mapped[int]     = mapped_column(ForeignKey("categories.id"), nullable=False)
    price:        Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    cost:         Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    stock:        Mapped[int]     = mapped_column(Integer, default=0)
    is_active:    Mapped[bool]    = mapped_column(Boolean, default=True)
    description:  Mapped[str]    = mapped_column(Text, nullable=True)
    created_at:   Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at:   Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    category:    Mapped["Category"]      = relationship("Category", back_populates="products")
    order_items: Mapped[list["OrderItem"]] = relationship("OrderItem", back_populates="product")
    reviews:     Mapped[list["Review"]]   = relationship("Review", back_populates="product")


class Order(Base):
    __tablename__ = "orders"

    id:               Mapped[int]         = mapped_column(Integer, primary_key=True, index=True)
    customer_id:      Mapped[int]         = mapped_column(ForeignKey("customers.id"), nullable=False, index=True)
    status:           Mapped[OrderStatus] = mapped_column(SAEnum(OrderStatus), default=OrderStatus.pending)
    total_amount:     Mapped[Decimal]     = mapped_column(Numeric(10, 2), nullable=False)
    discount_amount:  Mapped[Decimal]     = mapped_column(Numeric(10, 2), default=0)
    shipping_country: Mapped[str]         = mapped_column(String(100), nullable=False)
    created_at:       Mapped[datetime]    = mapped_column(DateTime, server_default=func.now(), index=True)
    updated_at:       Mapped[datetime]    = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    customer:    Mapped["Customer"]        = relationship("Customer", back_populates="orders")
    order_items: Mapped[list["OrderItem"]] = relationship("OrderItem", back_populates="order")


class OrderItem(Base):
    __tablename__ = "order_items"

    id:          Mapped[int]     = mapped_column(Integer, primary_key=True, index=True)
    order_id:    Mapped[int]     = mapped_column(ForeignKey("orders.id"), nullable=False, index=True)
    product_id:  Mapped[int]     = mapped_column(ForeignKey("products.id"), nullable=False)
    quantity:    Mapped[int]     = mapped_column(Integer, nullable=False)
    unit_price:  Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    line_total:  Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)

    order:   Mapped["Order"]   = relationship("Order", back_populates="order_items")
    product: Mapped["Product"] = relationship("Product", back_populates="order_items")


class Review(Base):
    __tablename__ = "reviews"

    id:          Mapped[int]      = mapped_column(Integer, primary_key=True, index=True)
    customer_id: Mapped[int]      = mapped_column(ForeignKey("customers.id"), nullable=False)
    product_id:  Mapped[int]      = mapped_column(ForeignKey("products.id"), nullable=False)
    rating:      Mapped[int]      = mapped_column(Integer, nullable=False)   # 1–5
    comment:     Mapped[str]      = mapped_column(Text, nullable=True)
    created_at:  Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    customer: Mapped["Customer"] = relationship("Customer", back_populates="reviews")
    product:  Mapped["Product"]  = relationship("Product", back_populates="reviews")
