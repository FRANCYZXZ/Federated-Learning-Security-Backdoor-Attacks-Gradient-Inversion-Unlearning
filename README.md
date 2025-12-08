# Federated Learning Security: Backdoor Attacks, Gradient Inversion & Unlearning

This repository provides a comprehensive framework for simulating and analyzing security threats in Federated Learning (FL).  
The project explores the interaction between two critical classes of attacks:

- **Integrity Attacks (Backdoor/Poisoning):** Injecting malicious patterns into client data to alter global model behaviour.  
- **Privacy Attacks (Gradient Inversion):** Reconstructing private client data from shared gradients or model updates.

Additionally, it introduces an **active defense strategy** based on *Machine Unlearning*, leveraging reconstructed images from Gradient Inversion to remove backdoor effects from the global model.

---

## Key Features

- **Complete FL Environment**  
  Simulation of a central server and multiple clients on CIFAR-10 or MNIST.

- **Backdoor Attacks**  
  Pixel-pattern poisoning with optional weight boosting to bypass FedAvg.

- **Robust Aggregation Rules**  
  Support for FL-Defender, FoolsGold, Krum, Trimmed Mean, and standard FedAvg.

- **Gradient Inversion (Privacy Breach)**  
  Integration of advanced algorithms to reconstruct training samples from model gradients/checkpoints.

- **"Train Once, Attack Many" Workflow**  
  Optimized checkpointing system separating heavy training steps from attack and analysis phases.

- **Machine Unlearning Pipeline**  
  Uses reconstructed images to fine-tune the model and remove implanted backdoors.

---

## Usage

The experiment workflow is divided into three modular phases.

---

### **Phase 1 — Federated Training**

This phase trains the global model (e.g., ResNet18) in a Federated Learning setup while simulating malicious peers performing a backdoor attack.  
At the end of the process, a `.t7` checkpoint containing the trained global model is saved.

```python
# reconstruction_only = False starts the federated training loop
run_exp(
    ...,
    rule='fedavg',
    attack_type='backdoor',
    reconstruction_only=False
)
```


### **Phase 2 — Reconstruction Attack (Gradient Inversion)**

In this phase, the previously saved model checkpoint is loaded and used to perform a **Gradient Inversion Attack**.  
The goal is to simulate a *curious server* or a malicious actor attempting to reconstruct the poisoned client image from the shared model updates.

This phase **does not perform any training**.  
Instead, it directly attacks the stored model parameters to extract the data used by the backdoored client.

Run the reconstruction with:

```python
# reconstruction_only = True skips training and performs only the reconstruction attack
run_exp(
    ...,
    reconstruction_only=True
)
```
After execution, the reconstructed images will be stored in:
```python
reconstructed_images/
```

These images are used in Phase 3 for verification and unlearning.

### **Phase 3 — Verification & Unlearning**

This phase evaluates the effectiveness of the backdoor attack and performs a **Machine Unlearning** procedure to remove the malicious behavior from the global model.

The process includes two key steps:

---

#### **1. Backdoor Verification**

The script first checks whether the reconstructed image (obtained in Phase 2) successfully triggers the targeted misclassification.  
This step confirms whether the backdoor is active within the global model.

---

#### **2. Corrective Unlearning**

If the attack is successful, the script performs a targeted fine-tuning operation using the reconstructed image as corrective data.  
This aims to:

- Suppress or eliminate the backdoor trigger  
- Restore the model's integrity  
- Ensure the global model no longer responds to the poisoned pattern  

Run the unlearning pipeline with:

```bash
python unlearning.py
```

The script outputs:

- Attack success status

- Updated model checkpoint without the backdoor

- Evaluation metrics before and after unlearning

This closes the loop by using privacy leakage (gradient inversion) as a defense mechanism against integrity attacks.

## References & Credits

This project is built by integrating and extending concepts, algorithms, and code from well-established research in Federated Learning security and Gradient Inversion.

---

### **Federated Learning Framework & Defense Methods**

The FL environment, peer management logic, and the FL-Defender defense mechanism are based on:

- **Repository:** FL-Defender  
- **Author:** Najeeb Jebreel  
- **Paper:** *FL-Defender: Combating Targeted Attacks in Federated Learning*

This project adapts and extends the original codebase to support:
- backdoor attack simulation  
- custom aggregation rules  
- gradient inversion integration  
- machine unlearning pipeline  

---

### **Gradient Inversion Algorithm**

The reconstruction pipeline for recovering images from gradients is based on the *invertinggradients* library:

- **Repository:** invertinggradients  
- **Author:** Jonas Geiping  
- **Paper:** *Inverting Gradients – How Easy Is It to Break Privacy in Federated Learning?*  
  NeurIPS 2020

This work provides the theoretical foundation and implementation strategy for privacy attacks against FL systems.

---
