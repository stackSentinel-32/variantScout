# VariantScout repository review

Reviewed 21 November 2026 against the working tree, including the local edit to `src/baseline.py`.

## Are we on the right track?

**Yes for a retrospective study of single-substitution tolerance. The app still needs correctness fixes before its rankings can carry the benchmark claim.** The evaluation ranks features within each assay, holds out entire UniProt IDs, and compares top-ten hits with each assay's own base rate. Keep that design.

I recalculated these means from the 20 Ridge rows in [loo_metrics.csv](../results/tables/loo_metrics.csv).

| Question | Ridge | Random |
|---|---:|---:|
| What fraction of the ten picks meet the assay's functional threshold? | 89.0% | 63.7% |
| What fraction reach or exceed the assay's 90th-percentile score? | 16.0% | 13.5% |

Mean Spearman is 0.4614. The two precision figures answer different user questions. Evidence for finding functional variants is stronger than evidence for finding the very highest-scoring variants. Ties at the threshold explain why the second random baseline can exceed 10%. The tables measure performance among experimentally observed singles. The app ranks all substitutions, applies a structural filter and caps picks per position. Its fallback, larger model option and double-mutant output have no corresponding held-out result in these tables.

## Fix first

| Priority | Location | Finding and consequence | What to do and how to check it |
|---|---|---|---|
| High | `app/app.py:1134` | Results exist only on the run where **Score mutations** returns true. A viewer selector or results slider triggers another Streamlit run, hits `st.stop()`, and loses the results view. | Store the completed result and submitted inputs in session state. Score once, change a viewer control, and verify that the results and downloads remain available without another scoring run. |
| High | `app/app.py:1178`, `:1246`, `:1255` | A failed structure lookup produces missing RSA. On a fresh feature table the missing values survive ranking and reach Ridge, which rejects NaN. Partial residue coverage can cause the same failure. Missing pLDDT is also labelled “≥ 70”. | Stop before scoring with a clear coverage message when required structural inputs are missing. Display unknown confidence explicitly. Test an unavailable structure and a structure missing one residue. |
| High | `src/esm.py:58`, `app/app.py:216`, `:1224` | ESM accepts an old matrix for a different same-length sequence under the same ID. The app's Parquet check verifies wild-type positions but ignores model revision, alignment, structure and substitution coverage. A second alignment can therefore inherit the first alignment's features. The existing positive cache test even accepts a table with one row per position and no required features. | Use content fingerprints and a schema version. Validate required columns, unique substitution keys and expected coverage. Reject legacy entries with unverifiable inputs. Use synthetic fixtures to test equal-length sequences, changed alignments and incomplete tables. |
| High | `src/baseline.py:23`, `:51`, `src/evaluate.py:61` | Baseline caches observed rows under UniProt ID. Two assays of one protein can cover different mutants. The left join silently introduces missing values on cache hits. The working-tree guard also reports the first ten input rows as `top10_fn` when the ranking feature is absent. Evaluation still indexes every required feature. | Require the assay identity, feature schema, unique keys and complete join coverage before accepting a cache. Mark an unavailable ranking metric as missing with a reason. Test partial overlap and missing columns, including the downstream evaluation call. |
| High | `app/app.py:1196`, `src/msa.py:11`, `src/structure.py:39` | Uploaded alignments discard the parsed query, and structures use the first chain's integer residue IDs. Neither app path verifies that these residues correspond to the submitted sequence. The MSA reader concatenates records before reshaping, so unequal widths can silently move characters between records when their total is divisible by the record count. | Validate each input record's width, the query identity, and the structure chain and residue mapping before combining inputs. Use mismatched and ragged synthetic fixtures. Show a clear error for ambiguous mappings. |
| High | `app/app.py:1199`, `:260`, `:1174` | All sessions write uploaded alignments to the same file. The current checkout retains a roughly 68 MB upload there. Interleaved requests can read another session's input. Unresolved identifier text also flows into local filenames. | Parse uploads from memory or use request-scoped temporary storage with cleanup. Use generated cache keys and constrain all file access to its intended directory. Check two overlapping uploads with different contents. |

## Evaluation and reporting

1. **Validate the reference comparison on matching rows.** `src/esm.py:118` expects a column called `score`. The local ProteinGym files use `ESM2_35M` and other model-specific columns. The helper also returns only the reference correlation. On matched single substitutions I independently obtained 0.07617 for GFP (1,084 rows) and 0.10817 for P53 (1,048 rows), agreeing with the saved table to its displayed precision. Accept an explicit reference column, check join cardinality, and report both correlations and coverage on the same rows. These two spot checks support the reported observations. They do not establish a general depth rule.
2. **Publish the scope of each metric.** `src/evaluate.py:78` takes the top ten measured rows directly. The interactive shortlist uses a different candidate population and filters (`app/app.py:496`, `:1333`). Label the current figures as retrospective benchmark results. Report the actual shortlist policy separately on held-out measured rows, with eligible counts and its matching random baseline. Label score-percentile tiers as ranks until calibration data support a confidence claim. Describe the double-mutant output as an unvalidated heuristic.
3. **Correct the README from the saved outputs.** LightGBM binary precision averages **89%**, while the README says 88%. The MSA implementation uses a fixed 0.5 pseudocount per residue type, excludes gaps from the denominator and uses natural logarithms. The README specifies a depth-dependent pseudocount and base-2 logs. Update the formulas and units. Remove the recommendation to use UBC9 as a held-out app example (`app/app.py:1295`), since it is in `STARTER_SET`.
4. **Keep the statistical unit consistent as coverage expands.** `evaluate_predictions` returns one row per assay. The bootstrap then samples those rows. Today's table has one assay per held-out protein. Adding a second assay for a protein would give it extra weight and treat related results as independent. Retain the UniProt ID through reporting and resample by protein. Save confidence intervals with the metrics, including the paired precision gain.
5. **Show the limits of the MSA-depth conclusion.** The plot title in `src/evaluate.py:259` states the conclusion unconditionally. The saved table includes a near-zero ESM result at Neff/L 168.82 as well as weak results at shallow depths. Show the paired signal difference across all assays and distinguish an observed subgroup from a rule for future proteins.
6. **Keep depth estimates tied to the actual alignment.** `app/app.py:1207` labels an upload with a reference depth by matching protein ID alone. `src/msa.py:54` scales subsampled effective counts by total/sample size. For 3,000 identical rows the formula yields 2 although there is one distinct cluster. Use the published value only for the matching reference alignment, label estimates, and add the identical-row case to tests. The reference test currently checks a metadata field rather than the estimator.

## Other user-visible failures and reproducibility gaps

- **Multi-record FASTA silently joins proteins.** `app/app.py:192` strips every header then concatenates all sequence lines. Require exactly one record while accepting wrapped lines. Test single-record, wrapped and multi-record input.
- **An empty selection crashes the attribution panel.** After filtering every candidate, `app/app.py:1385` builds an empty selector and indexes the first matching row. The fallback panels use the same pattern. Show an empty-state message and skip dependent controls. Check with a fixture where every position is excluded.
- **An incomplete benchmark can overwrite published results.** `src/baseline.py:77` skips missing alignments and writes the resulting subset over the normal table. `src/evaluate.py:50` also skips them. Preflight the expected cohort and write a run manifest before replacing results. Keep deliberately partial runs in a separate output directory. The saved model pickle also needs a training-data and feature-version manifest.
- **A clean install and a developer checkout behave differently.** Several tests require untracked datasets or model weights. Some skip missing data, others call `load_reference()` unconditionally. Move file-dependent checks into an explicit integration suite and keep small synthetic fixtures for offline unit tests. The Makefile hardcodes Windows executables even though README shows Bash usage. Docker uses `COPY . .` without a `.dockerignore`, allowing the local environment, raw data, cached uploads and Git history into the image. Exclude them and document the required data mount.

## Recommended order

1. Preserve the current tables as a reference. Fix cache identity, input validation, session isolation, missing-data handling and UI reruns. Use small synthetic inputs for regression checks.
2. Establish a repeatable benchmark run with a cohort manifest, reference checks, and separate reporting for the policy used by the app. Update the README from the resulting tables.
3. Revisit stronger accuracy claims and additional features after those two milestones. The available evidence already supports a narrower and useful portfolio result about retrospective tolerance ranking.

## Review coverage

Five selected offline tests passed in 53 seconds. These covered the three validation and cache tests, the small synthetic Neff case and the available-model configuration. The new failure cases above came from source inspection and artifact checks. The existing suite does not exercise most of them.

The review covered the 48 tracked files across the app, source, tests, configuration, documentation, tables and figures, plus the current baseline edit. All 21 tracked PNGs passed image-file integrity checks. I inventoried the local raw and processed data directories, inspected feature-cache schemas, checked that the 20 starter MSAs are present, and spot-checked local reference data. Third-party virtual-environment files, Git internals and individual records throughout the bulk datasets were outside the source review. The saved metrics were recalculated from the CSVs. The full model-training pipeline, live app and Docker build were not rerun.

## Copy-ready AI prompts, stage by stage

Use one stage per conversation or coding-agent task. Paste the shared prompt first, then the stage prompt. Give the model the current working tree and the previous stage's handoff. For a chat model without repository access, attach the issue file, the relevant source and tests, and any current diff for those files. Start Stage 1 even if some findings may already have been fixed.

### Shared prompt for every stage

```text
Work on the VariantScout repository. Read doc/repository-issues.md and the current implementation before editing. The issue report describes one snapshot. Verify each finding against the current code and use function names to locate code if line numbers have moved.

Complete only the stage I paste below. Preserve existing user edits, especially in src/baseline.py. Use temporary directories, synthetic fixtures and mocked external services for software regression checks. Keep published results and real user uploads intact. For scientific reporting, work from existing aggregate outputs or benign reporter data such as GFP. Keep the work focused on correctness, provenance and retrospective evaluation.

If you can edit and execute code, implement the change and run the checks. If you only have chat access, request the smallest missing set of files, supply a patch and exact verification commands, and label execution as pending. If a check fails, investigate the first relevant error and retry after a targeted fix. Report any remaining blocker and its impact.

End with a handoff I can paste into the next conversation. Include the stage number, changed files, commands actually run, observed results, skipped checks with reasons, and any decision the next stage depends on.
```

### Stage 1 — Establish a reproducible starting point

```text
Inspect the working-tree status, existing tests, Python environment and available data. Record the existing diff and checksums of the three tracked result tables in your handoff. Recalculate the two Ridge precision means and their random baselines from loo_metrics.csv. Identify the denominator and unit of averaging. Run the existing offline checks for input validation, cache validation, synthetic Neff and model configuration. List any tests that require local datasets or downloaded weights separately.

Finish when another developer can repeat the commands from the repository root, distinguish pre-existing edits from your work, and see which checks passed, failed or skipped. This stage is read-only.
```

### Stage 2 — Prevent stale caches and misleading baseline summaries

```text
Inspect score_sequence, score_assay, is_valid_feature_cache and load_baseline_models, including their callers. Make each cache validate the inputs its contents depend on. Distinguish a measured assay table from a complete app table. Validate required columns, unique row keys and coverage before reuse. Handle unverifiable legacy entries without deleting the user's original files. Ensure interrupted writes cannot appear as successful cache entries.

Use temporary fixtures to show that identical inputs reuse a valid entry, same-length changed sequences miss, changed alignment or structure inputs miss where relevant, and overlapping assays cannot borrow incomplete rows. Exercise the baseline path with the ranking feature missing. It must report an unavailable metric or a clear error instead of treating file order as a ranking. Include a regression check for the caller that consumes the feature table.
```

### Stage 3 — Validate submissions and isolate uploads

```text
Trace FASTA, alignment and structure inputs from the Streamlit form into their parsers. Accept a wrapped single FASTA record. Give a clear error for multiple records, unequal alignment widths, a mismatched query or an ambiguous chain or residue mapping. Keep uploaded contents request-local. Use generated internal filenames for caches. Tie any displayed reference depth to the actual reference alignment.

Use tiny invented records to exercise the real parsing functions. Include unequal record widths whose combined length is divisible by the record count, two concurrent uploads, and 3,000 identical alignment rows. For the last case verify that the reported effective count is one, or that an unsupported estimate is explicitly withheld. Report the user-visible message for each rejected input.
```

### Stage 4 — Keep the Streamlit results usable across reruns

```text
Follow main() from submit through result rendering. Keep submitted inputs and completed results in session state so changing a viewer control or results filter preserves the current result. Make edited but unsubmitted inputs distinguishable from the submitted result. Show a clear state for missing structural coverage, unknown confidence and zero eligible rows. Handle both full and fallback views.

Run a Streamlit interaction check with small mocked inputs. Submit once, change a viewer control, change a filter until the result is empty, then submit a different input. Verify the expensive calculation runs once per new submission, the displayed result belongs to the correct submission, and empty panels render cleanly. Also exercise an unavailable structure and a partially covered structure. Record which checks reached the actual UI.
```

### Stage 5 — Audit evaluation and protect published results

```text
Inspect build_dataset, run_loo, evaluate_predictions, bootstrap_ci and the output-writing entry points. Preserve within-assay normalization and holdouts by UniProt ID. Verify with a toy dataset containing two assays for one protein that it never appears in both train and test and that the uncertainty calculation resamples proteins. Use a tied-score fixture to check both precision definitions and their random expectations. State how eligible counts below ten are handled.

Use an isolated output directory to simulate missing inputs and an interrupted run. Verify the tracked tables stay byte-for-byte intact. Record cohort, exclusions and input versions per run. For the published-reference comparison, check the explicit model column, one-to-one row matching and matched-row count using GFP or a synthetic table. Report aggregate results. Identify which published metrics cover the interactive filters, fallback and other app modes, and mark any unmeasured mode clearly.
```

### Stage 6 — Make setup and tests repeatable

```text
Read requirements.txt, environment.yml, pytest.ini, Makefile, Dockerfile and the setup section of README.md together. Give a new contributor a working path to run the offline suite without benchmark downloads or model weights. Mark integration tests and document their real prerequisites. Make the documented commands agree with the supported operating systems. Exclude local environments, datasets, caches, uploads and Git metadata from the Docker build context, and document mounts for data-dependent runs.

Verify the documented offline command in an isolated environment where feasible. Check test collection for the integration suite. If Docker is available, build and smoke-test the documented entry point. State which platforms and commands you actually exercised. Include exact commands for any remaining verification.
```

### Stage 7 — Reconcile the README and sign off

```text
Review the final diff against doc/repository-issues.md and the earlier handoffs. Recalculate every headline number you retain from the saved tables. Check formulas against the current code. Distinguish binary functional hits, top-decile hits and unvalidated app modes. Check that examples described as held out are absent from the actual training cohort. Qualify MSA-depth conclusions to match the evidence. Treat score-percentile tiers as relative ranks unless a calibration result exists.

Run the offline regression suite, review the final working-tree status and compare published-table checksums with Stage 1. Explain any intended differences. End with a checklist mapping each original issue to fixed and verified, fixed but unverified, still open, or superseded. Cite a test, command output or code location for each status. State which retrospective claim the evidence supports and what remains before the interactive app can make the same claim.
```
