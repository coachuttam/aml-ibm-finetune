# SETUP_PROMPT.md — KYC Onboarding Assistant

Act as a Senior AI Engineer, Solution Architect, and Technical Lead.

I want to build a practical banking technology project called **KYC Onboarding Assistant**.

Your first task is to inspect the project directory, understand this brief, and create the foundational documentation. Do not start implementing the application until the documentation is reviewed.

The project must be beginner-friendly, technically credible, testable, and achievable as a focused three-day MVP.

# 1. Project objective

Build a human-in-the-loop KYC onboarding system.

A customer submits:
- One identity document (passport)
- One address proof (utility bill)
- An application form

The system extracts information, validates fields using deterministic Python logic, cross-checks documents, screens names against a synthetic sanctions/PEP watchlist, calculates rule-based risk, and drafts an onboarding note.

A human compliance officer makes the final decision.

Possible decisions:
- APPROVE
- REFER_FOR_ENHANCED_DUE_DILIGENCE
- REJECT

**Core principle:**

OCR extracts. Python validates. Rules calculate risk. The LLM explains. A human decides.

The LLM must never independently extract, validate, score, approve, or reject a customer.

# 2. MVP scope

Keep the first version achievable in approximately three days.

### Document types

1. Passport
2. Utility bill

Use only synthetic documents generated specifically for this project.

Never use real Aadhaar, PAN, passport, or customer identity documents.

### Fields

Extract and retain confidence scores for:

- Full name
- Date of birth
- Document number
- Address
- Issue date
- Expiry date

### Main workflow

1. Generate or upload synthetic documents.
2. Extract fields using OCR.
3. Validate fields using deterministic Python functions.
4. Verify passport MRZ check digits.
5. Cross-check identity, address, and application information.
6. Screen names against a synthetic sanctions/PEP watchlist.
7. Calculate a rule-based risk score.
8. Generate an explanatory onboarding note using an LLM.
9. Display the case in a Streamlit review interface.
10. Record the human decision and maintain an audit trail.

Low-confidence OCR, document mismatches, and potential watchlist matches must be routed for human review.

# 3. First implementation priority: Synthetic document generator

This is the first module to design and implement.

Create a synthetic document generation system that produces:

- Realistic-looking synthetic passports
- Synthetic utility bills
- Matching application forms
- A ground-truth JSON file containing the actual values used to generate each document

### Passport generator

Include:
- Synthetic full name
- Date of birth
- Passport number
- Nationality
- Issue date
- Expiry date
- MRZ lines

Generate valid MRZ check digits using the actual ICAO calculation rules.

Also generate selected intentionally invalid examples, such as incorrect check digits or expired documents, for testing.

### Utility bill generator

Include:
- Synthetic customer name
- Address
- Billing date
- Account/reference number
- Utility provider name

### Ground-truth evaluation

The generator must retain the correct field values independently of OCR output.

Use this ground truth to measure:
- Field-level exact-match accuracy
- Date extraction accuracy
- Document-number accuracy
- Name extraction accuracy
- Address extraction accuracy
- MRZ validation success

Keep generated documents and ground-truth data clearly separated.

The evaluation must compare OCR output against the original ground truth. Never use OCR output as its own reference.

This creates a measurable evaluation dataset rather than an OCR demonstration with unverified claims.

# 4. Technology stack

Keep the stack minimal.

Recommended initial stack:

- Python 3.12
- Tesseract OCR
- Pytesseract
- Pydantic
- Streamlit
- SQLite
- Pytest
- One LLM provider behind a simple replaceable interface

Do not introduce FastAPI in the MVP. Streamlit can call the Python workflow directly.

Docker and deployment are future enhancements, not initial requirements.

Start with Tesseract. Do not add PaddleOCR unless there is a demonstrated technical need.

Explain the purpose of every dependency and avoid unnecessary packages.

# 5. Core engineering modules

Design the application around simple, independently testable modules.

Suggested responsibilities:

- `document_generator.py` — creates synthetic documents and ground truth.
- `ocr_engine.py` — extracts text and field-level confidence.
- `schema.py` — validates structured data.
- `validators.py` — document-number, date, expiry, and MRZ validation.
- `matching.py` — cross-document fuzzy matching.
- `watchlist.py` — synthetic sanctions/PEP screening.
- `risk_engine.py` — deterministic rule-based risk calculation.
- `llm_explainer.py` — drafts plain-English onboarding notes.
- `audit.py` — records system events and human decisions.
- `case_store.py` — persists cases in SQLite.
- `workflow.py` — orchestrates the onboarding workflow.
- `app.py` — Streamlit review interface.

Treat this as a proposed structure, not a requirement to create every file immediately. Keep the actual structure minimal and explainable.

# 6. Reuse from previous AML project

I have previously developed an AML Alert Investigator project.

Identify reusable architectural patterns and components:

- `schema.py` — structured output validation
- `audit.py` — audit logging
- `case_store.py` — case persistence
- Six-step workflow pattern
- Human review application

Explain:
1. What can be reused directly.
2. What needs modification for KYC.
3. What must be implemented specifically for KYC.

Do not blindly copy code or create unnecessary dependencies between the projects.

The KYC project must remain independent.

# 7. Functional requirements

Document the following requirements in the PRD with unique IDs (FR-001, FR-002, etc.).

### Synthetic documents
- Generate synthetic passports and utility bills.
- Generate valid and intentionally invalid examples.
- Retain ground-truth field values.
- Support reproducible test datasets.

### OCR
- Extract the six required fields.
- Retain field-level confidence scores.
- Handle missing or unreadable fields.
- Route low-confidence fields for human review.

Define a named constant such as `OCR_CONFIDENCE_THRESHOLD`.

Document:
- The initial threshold value.
- Why that value was selected.
- How it will be calibrated using the synthetic evaluation dataset.
- The consequences of falling below the threshold.

Do not present an arbitrary threshold as scientifically established.

### Deterministic validation
- Validate document-number formats.
- Validate date formats and date logic.
- Detect expired documents.
- Verify passport MRZ check digits.
- Record each validation result.

### Cross-document matching
Compare names and addresses across:
- Passport
- Utility bill
- Application form

Use fuzzy matching with explicit scores.

Do not treat every name variation as a mismatch.

Preserve the matching score and the reason for the result.

### Watchlist screening
Use a synthetic sanctions/PEP dataset.

Return potential matches with:
- Candidate name
- Match score
- Relevant matching information

A potential match is not a confirmed sanctions or PEP hit.

Define a named constant such as `WATCHLIST_MATCH_THRESHOLD`.

Document:
- The matching algorithm chosen (for example Jaro-Winkler or token-set ratio) and why.
- The threshold value and its justification.
- How it is calibrated against the synthetic watchlist.
- The trade-off between false positives and missed matches at that value.

### Risk engine
Use deterministic rules only.

Consider:
- Potential PEP/watchlist match
- Country risk
- Occupation risk
- Document quality
- OCR confidence
- Document mismatch count

Document the scoring rules and thresholds.

### LLM explanation
The LLM may:
- Summarise extracted and validated information.
- Explain discrepancies.
- Draft an onboarding note.
- Explain why a case needs human review.

The LLM must not:
- Extract source fields.
- Change verified values.
- Calculate risk.
- Make compliance decisions.

### Human decision
The compliance officer can:
- Approve
- Refer for enhanced due diligence
- Reject

Record:
- Reviewer identifier
- Decision
- Timestamp
- Reason or comments

Do not allow a final case export before a human decision is recorded.

# 8. Documentation files

Create five core documentation files and one behaviour specification.

### A. `PRD.md`

Include:
- Overview
- Problem statement
- Goals and non-goals
- Target users
- User stories
- Functional requirements
- Non-functional requirements
- Workflow
- Input/output schemas
- Synthetic document generation
- OCR confidence handling
- Validation rules
- MRZ verification
- Cross-document matching
- Watchlist screening
- Risk scoring
- Human review
- Audit requirements
- Security and privacy
- Acceptance criteria
- MVP scope
- Future enhancements
- Known limitations

Requirements must be specific and testable.

### B. `AGENTS.md`

Create instructions for AI coding agents.

Include:
- Project purpose
- Architecture principles
- Coding standards
- Naming conventions
- Module organisation
- Testing rules
- Documentation rules
- Security requirements
- Error handling
- Dependency management
- Git workflow

Mandatory rules:

1. Every code change must include meaningful test coverage.
2. Run `pytest tests/ -q` after every code change.
3. Report passing and failing test counts.
4. Update the README status section after meaningful milestones.
5. Update `tests/BEHAVIOURS.md` when expected behaviour changes.
6. Never claim an unimplemented feature works.
7. Use synthetic documents only.
8. Never allow the LLM to make compliance decisions.
9. Never silently trust low-confidence OCR.
10. Keep the code readable and easy to explain.

### C. `architecture.md`

Include:
- System context
- High-level architecture
- Component responsibilities
- Data flow
- Workflow
- Module interactions
- Data models
- Storage design
- OCR integration
- Validation engine
- Risk engine
- LLM integration
- Audit flow
- Human decision flow
- Security boundaries
- Error handling

Include Mermaid diagrams for:
- System architecture
- End-to-end workflow
- Human decision

Explain each diagram in plain language underneath it.

### D. `design_system.md`

Define the design system for the compliance review interface.

It should read as a professional banking and compliance application, not a generic AI tool.

Include:
- Design principles
- Colour palette and typography
- Layout, spacing and navigation
- Tables, forms and buttons
- Status and risk indicators
- OCR confidence indicators
- Document review panel
- Evidence panel linking each extracted field to where it came from
- Audit timeline
- Case summary
- Human decision panel
- Accessibility guidelines

The interface must visually distinguish four kinds of information:

1. OCR-extracted (unverified, with confidence)
2. Deterministically validated (verified by Python)
3. Calculated risk (rules-based)
4. Human decisions

Never use colour alone to carry risk or status. Pair every colour with a label or icon.

### E. `README.md`

A professional GitHub README.

Include:
- What the project does and the problem it solves
- Key features
- Architecture overview with the workflow
- Technology stack
- Project structure
- Installation, including the Tesseract system dependency
- Environment configuration and `.env.example`
- How to generate synthetic documents
- How to run the application and the tests
- An example end-to-end walkthrough using synthetic data
- OCR accuracy results measured against ground truth
- Security and privacy statement
- **Current status**: what is built, what is not
- Known limitations
- Roadmap

Do not claim any feature works before it is implemented and tested. The status section replaces a separate progress file; keep it current.

### F. `tests/BEHAVIOURS.md`

A behaviour specification that drives the test suite. Each scenario states the input, the expected system behaviour and the expected routing decision.

Cover:
- Valid passport, all fields high confidence
- Invalid passport number format
- Incorrect MRZ check digit
- Expired document
- OCR confidence below threshold on one field
- OCR confidence below threshold on several fields
- Name variation that should match (initials, reordering, middle name)
- Name mismatch that should not match
- Address mismatch between documents
- Potential watchlist match above threshold
- Near-threshold watchlist match
- No watchlist match
- High-risk case from accumulated rules
- Human referral for enhanced due diligence
- Human approval
- Human rejection
- Export blocked before a human decision
- Audit log records every step and the before-and-after state of a decision

# 9. Code readability

This is also a learning project. I must be able to explain every line.

- Simple functions over abstractions; one job per function, about twenty lines
- Meaningful names, no abbreviations a newcomer would have to guess
- Type hints throughout; Pydantic models where structured validation helps
- Docstrings on every module and public function, in plain English
- Comments explain why, never what
- No magic numbers: every threshold is a named constant with a one-line reason
- Business rules stay out of OCR and LLM code
- Each module testable on its own

When explaining an implementation decision, cover what it does, why it is needed, how it works, and a small example.

# 10. Security and privacy

- Synthetic documents only; never real Aadhaar, PAN, passport or customer data
- Secrets in `.env`, with a committed `.env.example`; never commit keys
- Never send real customer documents to an external LLM provider
- Watchlist hits are potential matches, never confirmed ones
- Audit every validation result, risk calculation and human decision
- Block export of a final decision or case report until a human has decided
- A system recommendation is never presented as a regulatory or legal determination

# 11. Project structure

Propose a minimal, beginner-friendly folder structure. Do not create empty modules or folders for files that do not exist yet. Explain the purpose of each folder and important file.

# 12. Working instructions

Before creating any file:

1. Inspect the project directory and identify anything that should be preserved.
2. Propose the folder structure.
3. List your assumptions and any decision that needs my approval.

Then create the documentation files above.

**Do not implement the application. Do not install dependencies. Do not create placeholder code.**

After the documentation exists:

1. Check the documents are consistent with each other.
2. Check every requirement is testable.
3. Check the architecture matches the PRD.
4. Check the README claims nothing that is not built.
5. Summarise the files created.
6. List decisions needing my approval.
7. Recommend the first implementation milestone.

Prioritise correctness, explainability, auditability and readable code over feature count.
