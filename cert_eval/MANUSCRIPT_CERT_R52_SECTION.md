## External Benchmark Evaluation on CERT r4.2

To address concerns regarding the limited scale of the Mesa-only simulation, we conducted an additional external benchmark evaluation using CERT r4.2. CERT r4.2 is not a smart-building or BMS dataset; therefore, it is not used as direct operational validation for municipal facilities. Instead, it provides a larger multi-modal insider-threat benchmark (~1,000 users across logon, device, file, web, and email modalities, with three labeled insider scenarios) for evaluating whether the proposed layered SIEM, trust-adaptive scoring, and evidence-gated confirmation logic generalize beyond the custom simulation.

We selected r4.2 over the larger r5.2 release after r5.2's 14+ GB HTTP log exceeded the memory budget of our evaluation environment. r4.2 shares the same column schema, ground-truth answer format, and insider scenarios at a more tractable scale, so the same evaluation pipeline applies without changes to the detection logic.

### Limitations

Although CERT r4.2 improves the scale and benchmark comparability of the evaluation, it remains a synthetic insider-threat dataset and does not fully reproduce the semantics of BMS control-plane activity, HVAC scheduling, access-control operations, or vendor maintenance workflows. Therefore, the CERT evaluation should be interpreted as external insider-threat benchmark evidence rather than real-world smart-building deployment validation.
