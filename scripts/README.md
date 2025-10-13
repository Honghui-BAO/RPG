# RPG Scripts

This directory contains shell scripts for training and testing RPG models.

## 📜 Available Scripts

### RPG (Original Autoregressive Model)

- **[run_beauty.sh](run_beauty.sh)** - Train and test RPG on Beauty dataset
  ```bash
  ./scripts/run_beauty.sh
  ```

### LLaDA (Diffusion Model)

- **[train_llada.sh](train_llada.sh)** - Train LLaDA diffusion model
  ```bash
  ./scripts/train_llada.sh
  ```

- **[test_llada.sh](test_llada.sh)** - Test LLaDA model (without graph)
  ```bash
  # Remember to set checkpoint path in the script
  ./scripts/test_llada.sh
  ```

## 🔧 Usage

All scripts should be run from the project root directory:

```bash
cd /path/to/RPG
./scripts/train_llada.sh
```

## ⚙️ Configuration

Scripts use the following proxy settings (modify if needed):
```bash
export http_proxy=http://oversea-squid1.jp.txyun:11080
export https_proxy=http://oversea-squid1.jp.txyun:11080
```

## 📝 Creating New Scripts

When creating new scripts:
1. Add them to this directory
2. Make them executable: `chmod +x scripts/your_script.sh`
3. Document them in this README
4. Run from project root directory


