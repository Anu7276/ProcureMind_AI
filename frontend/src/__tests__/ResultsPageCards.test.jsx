import { describe, it, expect } from 'vitest'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import {
  RecommendationCard,
  CertificationSection,
  VersionCheckSection,
  AmendmentsSection,
  VerificationSection,
} from '../pages/ResultsPage.jsx'

describe('Standard Result Card Consistent Section Structure', () => {
  const cardWithData = {
    key: 'IS 1786',
    is_code: 'IS 1786:2008',
    title: 'High strength deformed steel bars and wires for concrete reinforcement',
    status: 'ACTIVE',
    confidence: 0.95,
    match_strength: 0.95,
    coverage: 0.9,
    reasoning: 'Matches requirement for TMT rebars.',
    certification: {
      mandatory: true,
      schemes: ['Scheme-I / ISI Mark'],
      scheme_name: 'Scheme-I / ISI Mark',
      scheme_code: 'Scheme-I',
      gazette_reference: 'S.O. 3450(E)',
      penalty: 'Section 29 of BIS Act 2016',
      evidence_source: 'qco_orders.json',
    },
    version_check: {
      current_edition_year: 2008,
      cited_year: 2008,
      is_current: true,
      status: 'ACTIVE',
      successors: [],
      messages: ['Tender cited year 2008 matches current active edition.'],
    },
    amendments: {
      status: 'listed',
      entries: [
        { amendment_no: 1, year: 2020, note: 'Clarification on chemical tolerances' },
      ],
    },
    verification: {
      level: 'verified_multi_source',
      flags: ['multi_source'],
    },
    related_standards: [],
    evidence_sources: ['IS 1786:2008, BIS'],
  }

  const cardWithoutData = {
    key: 'IS 158',
    is_code: 'IS 158',
    title: 'Ready mixed paint, brushing, bituminous, black, lead-free',
    status: 'ACTIVE',
    confidence: 0.65,
    match_strength: 0.65,
    coverage: 0.5,
    reasoning: 'Matches general bitumen paint criteria.',
    certification: {
      mandatory: null,
      schemes: [],
      scheme_name: null,
      scheme_code: null,
      gazette_reference: null,
      penalty: null,
      evidence_source: null,
    },
    version_check: {
      current_edition_year: null,
      cited_year: null,
      is_current: null,
      status: 'ACTIVE',
      successors: [],
      messages: ['Edition year not available in dataset'],
    },
    amendments: {
      status: 'not_available_in_dataset',
      entries: [],
    },
    verification: {
      level: 'single_source_unconfirmed',
      flags: [],
    },
    related_standards: [],
    evidence_sources: [],
  }

  it('renders all 4 sections for both cards in fixed order (Certification, Version Check, Amendments, Verification)', () => {
    const htmlWithData = renderToStaticMarkup(React.createElement(RecommendationCard, { rec: cardWithData, rank: 0 }))
    const htmlWithoutData = renderToStaticMarkup(React.createElement(RecommendationCard, { rec: cardWithoutData, rank: 0 }))

    for (const [name, html] of [['Card With Data', htmlWithData], ['Card Without Data', htmlWithoutData]]) {
      expect(html).toContain('data-section="certification"')
      expect(html).toContain('data-section="version-check"')
      expect(html).toContain('data-section="amendments"')
      expect(html).toContain('data-section="verification"')

      const certIdx = html.indexOf('data-section="certification"')
      const verCheckIdx = html.indexOf('data-section="version-check"')
      const amendIdx = html.indexOf('data-section="amendments"')
      const verifIdx = html.indexOf('data-section="verification"')

      expect(certIdx).toBeLessThan(verCheckIdx)
      expect(verCheckIdx).toBeLessThan(amendIdx)
      expect(amendIdx).toBeLessThan(verifIdx)
    }
  })

  it('renders neutral grey style and "Not available / Not confirmed" text on unconfirmed card without confident claims', () => {
    const htmlWithoutData = renderToStaticMarkup(React.createElement(RecommendationCard, { rec: cardWithoutData, rank: 0 }))

    expect(htmlWithoutData).toContain('Not confirmed')
    expect(htmlWithoutData).toContain('Not available in the dataset')
    expect(htmlWithoutData).toContain('Edition year not available in dataset')
    expect(htmlWithoutData).toContain("This standard doesn&#x27;t have this information verified in our current dataset — check the official BIS source.")
    expect(htmlWithoutData).not.toContain('Active Edition (')
  })

  it('renders confident confirmed badges and amendment list on card with full data', () => {
    const htmlWithData = renderToStaticMarkup(React.createElement(RecommendationCard, { rec: cardWithData, rank: 0 }))

    expect(htmlWithData).toContain('Mandatory Certification')
    expect(htmlWithData).toContain('Active Edition (2008)')
    expect(htmlWithData).toContain('1 Amendment(s) Listed')
    expect(htmlWithData).toContain('Clarification on chemical tolerances')
  })
})
