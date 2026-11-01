# Architecture

```
human sources -> ingestion -> human data lake
                                 |         \
                                 |          \-> reserve pool (mining only)
                                 v
                        synthetic mirror engine
                                 v
                          training dataset
                                 v
                         model training (FSDP / DeepSpeed / Ray)
                                 v
                          evaluation lab (R1..R5 + RAID / MAGE / HC3)
                                 v
                           failure atlas  <----------------+
                                 v                         |
                        targeted data generation           |
                                 v                         |
                          next training cycle              |
                                 v                         |
                            release gate                   |
                                 v                         |
                        production API + UI                |
                                 v                         |
                    monitoring -> verified feedback -------+
```

The loop is the project. Everything else is plumbing that makes the loop measurable.

**Implementation status:** this diagram describes the intended research loop, not a fully
automated production feedback service. Ingestion, generation, training, evaluation, mining,
and the release-gate primitives exist. The deployed FastAPI app does not yet persist user
feedback or connect drift alerts and registry promotion to the mining DAG; feedback and
drift modules are standalone components. Treat the online monitoring-to-atlas edge as
partial until that integration is measured end to end.

## Why the loop, specifically

A conventional detector is trained once on whatever synthetic data was convenient and
then degrades as new models appear. The claim under test here is that choosing *which*
synthetic data to generate, based on where the current model actually fails, buys more
robustness per example than generating more data at random. That is why the Failure
Atlas sits between evaluation and generation instead of being a debugging convenience.

See `docs/data_spec_v1.md` for the frozen data contract and `docs/evidence.md` for
the infrastructure map.
