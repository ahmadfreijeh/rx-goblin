# My ML Learning Book

> Personal reference book. Command: "update book" to add new concepts.

---

## Table of Contents
1. [Transfer Learning](#transfer-learning)

---

## Transfer Learning

Taking a model that already learned something and redirecting it to your new task instead of training from scratch.

**The two parts of any model:**
- **Backbone** → extracts features from input (the eyes)
- **Head** → makes the final decision (the brain)

---

**Frozen Backbone**
- Backbone locked, only head gets trained
- Use when: new data is similar to original training data
- Example: ResNet50 → banana ripeness

```
Input → Backbone (frozen) → Head (trained) → Output
```

---

**Full Fine-tuning**
- Everything gets trained, backbone + head
- Use when: lots of data + strong GPU
- Risk: expensive, slow, catastrophic forgetting

```
Input → Backbone (trained) → Head (trained) → Output
```

---

**LoRA**
- Backbone frozen, tiny matrices injected alongside it and trained (~1% of parameters)
- Use when: small dataset, limited hardware
- Nearly same quality as full fine-tuning at 10% of the cost

```
Input → Backbone (frozen) + LoRA matrices (trained) → Head (trained) → Output
```

---

**When to use what:**
```
Data similar to original + simple task  → Frozen Backbone
Lots of data + strong GPU               → Full Fine-tuning
Small data + limited hardware           → LoRA
```
