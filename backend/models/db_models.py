"""
SQLAlchemy ORM models — mirrors the PostgreSQL schema in init.sql exactly.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Standard(Base):
    __tablename__ = "standards"

    id: int = Column(Integer, primary_key=True)
    key: str = Column(String(150), unique=True, nullable=False, index=True)
    display_code: str = Column(String(150), nullable=False, index=True)
    edition_year: Optional[int] = Column(Integer)
    title: str = Column(Text, nullable=False)
    scope: Optional[str] = Column(Text)
    status: str = Column(String(50), nullable=False, default="ACTIVE", index=True)
    category: Optional[str] = Column(String(100), index=True)
    subcategory: Optional[str] = Column(String(100))
    compliance_mandatory: Optional[bool] = Column(Boolean)
    verification_level: Optional[str] = Column(String(50), index=True)
    flags: List[str] = Column(JSONB, nullable=False, default=list)
    has_full_text: bool = Column(Boolean, nullable=False, default=False)
    superseded_by: List[str] = Column(JSONB, nullable=False, default=list)
    keywords: List[str] = Column(JSONB, nullable=False, default=list)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())


class QCOOrder(Base):
    __tablename__ = "qco_orders"

    id: int = Column(Integer, primary_key=True)
    qco_id: str = Column(String(100), unique=True, nullable=False)
    product: Optional[str] = Column(Text)
    standard_key: Optional[str] = Column(String(150), index=True)
    ministry: Optional[str] = Column(Text)
    enforcement_status: Optional[str] = Column(String(100))
    effective_date: Optional[datetime] = Column(Date)
    certification_required: Optional[str] = Column(Text)
    scope: Optional[str] = Column(Text)
    penalty: Optional[str] = Column(Text)
    gazette_reference: Optional[str] = Column(Text)
    origin: Optional[str] = Column(Text)
    verification: Optional[str] = Column(Text)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())


class CertificationScheme(Base):
    __tablename__ = "certification_schemes"

    id: int = Column(Integer, primary_key=True)
    scheme_code: str = Column(String(50), unique=True, nullable=False)
    name: str = Column(String(200), nullable=False)
    popular_name: Optional[str] = Column(String(100))
    description: Optional[str] = Column(Text)
    statutory_basis: Optional[str] = Column(Text)
    symbol: Optional[str] = Column(Text)
    applicable_sectors: List[str] = Column(JSONB, nullable=False, default=list)
    lead_time_weeks: Optional[int] = Column(Integer)
    is_mandatory_for_qco: bool = Column(Boolean, nullable=False, default=False)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())


class ProductRule(Base):
    __tablename__ = "product_rules"

    id: int = Column(Integer, primary_key=True)
    product: str = Column(Text, nullable=False, index=True)
    category: Optional[str] = Column(String(100))
    subcategory: Optional[str] = Column(String(100))
    state: Optional[str] = Column(String(100))
    standard_key: Optional[str] = Column(String(150), index=True)
    standard_raw: Optional[str] = Column(String(150))
    context: Optional[str] = Column(Text)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())


class RecommendationLog(Base):
    __tablename__ = "recommendation_log"

    id: int = Column(Integer, primary_key=True)
    audit_id: uuid.UUID = Column(
        UUID(as_uuid=True), unique=True, nullable=False, index=True
    )
    query_text: str = Column(Text, nullable=False)
    input_type: str = Column(String(50), nullable=False, default="text")
    structured_requirement: Optional[Dict[str, Any]] = Column(JSONB)
    returned_codes: List[str] = Column(JSONB, nullable=False, default=list)
    pipeline_warnings: List[str] = Column(JSONB, nullable=False, default=list)
    top_confidence: Optional[float] = Column(Numeric(4, 3))
    user_feedback: Optional[str] = Column(Text)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())
