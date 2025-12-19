# Examples

## Running Examples

Using hatch:

```bash
hatch run dev:train model_name=gpt2
```

Or directly:

```bash
cd examples
python ../fkat/train.py -cd conf -cn hf model_name=gpt2
```

## Jupyter Notebook

```bash
hatch run dev:notebook
```
