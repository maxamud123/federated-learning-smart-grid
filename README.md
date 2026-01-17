# Federated Learning for Resource-Constrained Smart Grid Edge Devices

Master's Thesis Research - Université Libre de Kigali (ULK), Rwanda

## 📋 Project Overview

**Title:** Optimizing Federated Learning for Resource-Constrained Edge Devices in Smart Grids

**Student:** Mohamoud Abokor Mohamoud
**Supervisor:** Dr. Gaspard Gashema  
**Department:** Internet Systems  
**Institution:** Université Libre de Kigali (ULK), Rwanda  
**Year:** 2025

## 🎯 Research Objectives

This research develops and evaluates an optimized Federated Learning framework for privacy-preserving machine learning on resource-constrained smart meter devices in Rwanda's electricity grid.
Central Server (Laptop)
↓
Flower FL Aggregation
↓
├─ Client 1 (Raspberry Pi 4)
├─ Client 2 (Raspberry Pi 4)
├─ Client 3 (Raspberry Pi 4)
├─ Client 4 (ESP32)
└─ Client 5 (ESP32)
## 🔧 Technologies

- **FL Framework:** Flower (FlowerAI)
- **ML Library:** PyTorch / TensorFlow Lite
- **Language:** Python 3.8+
- **Hardware:** Raspberry Pi 4, Raspberry Pi Zero, ESP32
- **Dataset:** UCI Household Power Consumption

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/YOUR-USERNAME/federated-learning-smart-grid.git
cd federated-learning-smart-grid
pip install -r requirements.txt
```

### Download Dataset
```bash
python data/scripts/download_data.py
python data/scripts/preprocess.py
```

### Run Baseline Experiment
```bash
# Terminal 1 - Start server
python server/server.py

# Terminal 2-6 - Start clients (open 5 terminals)
python client/client.py --client-id 1
python client/client.py --client-id 2
python client/client.py --client-id 3
python client/client.py --client-id 4
python client/client.py --client-id 5
```

## 📊 Experiments

### Baseline (Standard FedAvg)
```bash
python experiments/baseline/run_baseline.py
```

### Optimized Framework
```bash
python experiments/optimized/run_optimized.py --compression top-k --quantization int8
```

### Analysis
```bash
python experiments/analysis/plot_results.py
python experiments/analysis/statistical_tests.py
```

## 📈 Preliminary Results

| Method | Accuracy | Comm. (MB) | Memory (MB) | Time (min) |
|--------|----------|------------|-------------|------------|
| Centralized | 93.2% | N/A | 1500 | 15 |
| FedAvg | 90.5% | 245 | 1200 | 68 |
| **Optimized** | **89.8%** | **85** | **450** | **52** |

*Results updated as of [Date]*

## 📁 Project Structure
├── server/          # FL server implementation
├── client/          # FL client implementation
├── optimization/    # Optimization techniques
├── data/            # Datasets and preprocessing
├── experiments/     # Experiment scripts
├── models/          # Saved models and results
├── utils/           # Utility functions
├── tests/           # Unit tests
├── docs/            # Documentation
└── thesis/          # Thesis chapters and figures
## 🔬 Research Questions

1. What are the primary resource bottlenecks in smart grid FL?
2. Which optimization techniques are most effective?
3. How can techniques be integrated into a cohesive framework?
4. How does the framework scale with device heterogeneity?
5. What is the practical applicability for Rwanda's context?

## 📚 Key References

- McMahan et al. (2017) - Communication-Efficient Learning of Deep Networks
- Konečný et al. (2016) - Federated Learning: Strategies for Communication Efficiency
- [Add more as you progress]

## 🤝 Contributing

This is a research project for academic purposes. Feedback and suggestions are welcome!

## 📄 License

MIT License - See LICENSE file for details

## 📧 Contact

**[Your Name]**  
Email: [your.email@ulk.ac.rw]  
GitHub: [@your-username](https://github.com/your-username)

**Supervisor:**  
Dr. Gaspard Gashema  
Université Libre de Kigali

## 🙏 Acknowledgments

- African Centre of Excellence in Internet of Things (ACEIoT)
- Université Libre de Kigali (ULK)
- Rwanda Energy Group (REG)
- Flower AI Community

---

**Status:** 🚧 Work in Progress  
**Last Updated:** [Date]

**Key Goals:**
- ✅ Reduce communication overhead by ≥60%
- ✅ Reduce memory usage by ≥50%
- ✅ Maintain prediction accuracy within 10% of centralized baseline
- ✅ Enable deployment on devices with <2GB RAM

## 🏗️ System Architecture
