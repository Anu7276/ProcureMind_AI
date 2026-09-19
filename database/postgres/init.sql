-- ═══════════════════════════════════════════════════════════════
-- BIS Standards Recommendation Engine — PostgreSQL Schema
-- Auto-executed by Docker on first container start.
-- ═══════════════════════════════════════════════════════════════

-- ── Standards metadata ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS standards (
    id                  SERIAL PRIMARY KEY,
    key                 VARCHAR(150) UNIQUE NOT NULL,
    display_code        VARCHAR(150) NOT NULL,
    edition_year        INTEGER,
    title               TEXT NOT NULL,
    scope               TEXT,
    status              VARCHAR(50)  NOT NULL DEFAULT 'ACTIVE',
    category            VARCHAR(100),
    subcategory         VARCHAR(100),
    compliance_mandatory BOOLEAN,
    verification_level  VARCHAR(50),
    flags               JSONB        NOT NULL DEFAULT '[]',
    has_full_text       BOOLEAN      NOT NULL DEFAULT FALSE,
    superseded_by       JSONB        NOT NULL DEFAULT '[]',
    keywords            JSONB        NOT NULL DEFAULT '[]',
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_standards_key             ON standards(key);
CREATE INDEX IF NOT EXISTS idx_standards_status          ON standards(status);
CREATE INDEX IF NOT EXISTS idx_standards_category        ON standards(category);
CREATE INDEX IF NOT EXISTS idx_standards_verification    ON standards(verification_level);
CREATE INDEX IF NOT EXISTS idx_standards_display_code    ON standards(display_code);
-- Full-text search index on title for keyword fallback in Node03
CREATE INDEX IF NOT EXISTS idx_standards_title_fts
    ON standards USING gin(to_tsvector('english', title));

-- ── QCO (Quality Control Orders) ─────────────────────────────
CREATE TABLE IF NOT EXISTS qco_orders (
    id                   SERIAL PRIMARY KEY,
    qco_id               VARCHAR(100) UNIQUE NOT NULL,
    product              TEXT,
    standard_key         VARCHAR(150),
    ministry             TEXT,
    enforcement_status   VARCHAR(100),
    effective_date       DATE,
    certification_required TEXT,
    scope                TEXT,
    penalty              TEXT,
    gazette_reference    TEXT,
    origin               TEXT,
    verification         TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_qco_standard_key ON qco_orders(standard_key);

-- ── Certification schemes ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS certification_schemes (
    id                   SERIAL PRIMARY KEY,
    scheme_code          VARCHAR(50) UNIQUE NOT NULL,
    name                 VARCHAR(200) NOT NULL,
    popular_name         VARCHAR(100),
    description          TEXT,
    statutory_basis      TEXT,
    symbol               TEXT,
    applicable_sectors   JSONB NOT NULL DEFAULT '[]',
    lead_time_weeks      INTEGER,
    is_mandatory_for_qco BOOLEAN NOT NULL DEFAULT FALSE,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Product certification rules ───────────────────────────────
CREATE TABLE IF NOT EXISTS product_rules (
    id           SERIAL PRIMARY KEY,
    product      TEXT NOT NULL,
    category     VARCHAR(100),
    subcategory  VARCHAR(100),
    state        VARCHAR(100),      -- e.g. "Mandatory Certification"
    standard_key VARCHAR(150),
    standard_raw VARCHAR(150),
    context      TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_product_rules_standard_key ON product_rules(standard_key);
CREATE INDEX IF NOT EXISTS idx_product_rules_product      ON product_rules(product);

-- ── Recommendation audit log ──────────────────────────────────
-- Written by Node05 on every /recommend call.
CREATE TABLE IF NOT EXISTS recommendation_log (
    id                     SERIAL PRIMARY KEY,
    audit_id               UUID UNIQUE NOT NULL,
    query_text             TEXT NOT NULL,
    input_type             VARCHAR(50)  NOT NULL DEFAULT 'text',
    structured_requirement JSONB,
    returned_codes         JSONB        NOT NULL DEFAULT '[]',
    pipeline_warnings      JSONB        NOT NULL DEFAULT '[]',
    top_confidence         NUMERIC(4,3),
    user_feedback          TEXT,
    created_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rec_log_audit_id   ON recommendation_log(audit_id);
CREATE INDEX IF NOT EXISTS idx_rec_log_created_at ON recommendation_log(created_at DESC);
