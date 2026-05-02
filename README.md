# NOVA-GUARD

Physics-Aware Synthetic Cyber-Physical Attack Generation and Anomaly Detection in Satellite Telemetry

---

## Overview

NOVA-GUARD is an end-to-end framework for detecting stealthy cyber-physical attacks in satellite telemetry systems. The system combines a physics-constrained conditional generative adversarial network (GAN) for synthetic attack generation with an autoencoder-based anomaly detector, wrapped in a production-grade Model Context Protocol (MCP) architecture with agentic AI reasoning.

The framework addresses two critical challenges in spacecraft cybersecurity: the absence of labelled attack data for supervised training, and the difficulty of detecting coordinated multi-channel anomalies that evade conventional threshold-based monitors.

---

## Architecture

The system is organised into five layers:

**Layer 1: Data Acquisition and Preprocessing**
- NASA SMAP telemetry ingestion
- Missing value imputation, outlier removal, noise suppression
- Feature engineering (statistical, temporal, physics-based)
- Sliding window generation (50 x 25)

**Layer 2: Physics-Aware Generative Model**
- Conditional GAN with system state conditioning
- Physics constraint loss (energy conservation, thermal consistency, power balance, rate limits, cross-sensor correlation)
- Six attack scenario classes: stealth drift, coordinated manipulation, false data injection, bias injection, stochastic perturbation, multi-sensor composite

**Layer 3: Anomaly Detection Framework**
- Fully connected autoencoder (1250 -&gt; 128 -&gt; 64 -&gt; 32 bottleneck)
- Unsupervised training on normal data only
- Reconstruction error-based anomaly scoring
- Percentile-based threshold optimization

**Layer 4: Evaluation and Validation**
- Precision, Recall, F1-score, ROC-AUC, PR-AUC
- Ablation studies and baseline comparisons
- Cross-validation and rolling-origin validation

**Layer 5: Outputs and Deployment**
- Real-time anomaly detection and alerting
- Interactive diagnostic reports
- Mission Control dashboard
- Edge deployment support






