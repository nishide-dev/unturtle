---
name: Feature Request
about: New features, model support, dLLM algorithms, or ideas
title: "[Feature] "
labels: feature request
assignees: ''

---

### Is this related to an existing problem?
<!-- A clear description of what the problem is. -->

### Proposed solution
<!-- What would you like to happen? -->

### For new dLLM model support, have you tried:
```python
from unturtle import FastModel
model, tokenizer = FastModel.from_pretrained("your-model")

# For diffusion LM training:
from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments, LinearAlphaScheduler
```

### Additional context
<!-- Any references, papers, or related implementations (e.g. LLaDA, MDLM, d1). -->
