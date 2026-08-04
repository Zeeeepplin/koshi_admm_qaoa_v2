# Eastern Nepal test-system provenance audit — 2026-07-25

## Source checked

Primary source: Nepal Electricity Authority, *Transmission Directorate
Year Book 2081/82*, official PDF:

`https://pmitd.nea.org.np/uploads/shares/annual_report/Transmission_Directorate_2082.pdf`

The audit distinguishes reported assets and project status from model
assumptions. The resulting 16-bus case is a source-informed research test
system; it is not an as-operated NEA network model or utility single-line
record.

## Verified source facts

- PDF page 15 (printed page 13) reports the 35 km
  Dhungesanghu–Basantapur line, a 132/33 kV 2×15 MVA Dhungesanghu
  substation bank, commissioning of the KC4 second circuit in 2025, and
  the 73 MW Sanima Middle Tamor plant.
- PDF page 21 (printed page 19) lists the planned
  Amarpur–Dhungesanghu 19.13 km double-circuit line, with expected
  completion in FY 2082/83 and construction status at the source date.
- PDF pages 29–30 (printed pages 27–28) describe KC1 as 107 km of
  double-circuit towers initially strung with one circuit, with KC5
  adding the second circuit, and identify conductor types. They do not
  provide every segment length used by this model.
- PDF page 41 (printed page 39) reports Inaruwa transformer-bank
  capacities of 3×315 MVA at 400/220 kV and 2×160 MVA at 220/132 kV.
- PDF pages 85–88 contain the existing-line tables used as a secondary
  cross-check on route and voltage-class descriptions.

## Corrections made

- The Inaruwa equivalents were corrected to 945 MVA (3×315) and 320 MVA
  (2×160).
- A previously asserted 2×15 MVA 220/132 kV interface was removed. The
  source's 2×15 MVA value belongs to the 132/33 kV Dhungesanghu bank.
- The model now treats the Amarpur–Dhungesanghu 220/132 kV interface as
  an explicit hypothetical 100 MVA research assumption, while retaining
  the reported 19.13 km planned tie.
- KC1 circuit B and the Amarpur tie are labelled planned/prospective,
  not in-service assets for the source year.
- The 73 MW Sanima Middle Tamor injection is source-reported. Tumlingtar
  and Amarpur aggregate injections, nodal loads and reactive demands are
  operating-point assumptions.

## Values that remain engineering assumptions

Unless the generated branch-provenance table says otherwise, branch
segment lengths, conductor-derived impedances, MVA ratings, transformer
short-circuit reactances, unit taps, and the aggregated operating point
are research assumptions. The paper now states these limitations
explicitly. Operational use would require utility-validated
SCADA/EMS/planning data, protection status, tap positions, and a dated
network snapshot.

## Scenario interpretation

The branch-3 Basantapur–Inaruwa circuit-A outage is a predeclared
hypothetical deterministic N−1 stress case. It is meaningful as removal
of one circuit on a major corridor in the model, but no historical
operational record was found or claimed. Results must therefore be
reported as test-system evidence, never as reconstruction of an actual
NEA event.
