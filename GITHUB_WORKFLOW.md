# LangGraph GitHub Workflow - Automation Guide

## 📂 Repository Structure

```
langgraph-git/
├── GITHUB_WORKFLOW.md          # This file - workflow documentation
├── push_to_github.sh            # Automation script for commits & push
├── quick_push.sh                # Quick commit and push script
├── .gitignore                   # Git ignore patterns
├── README.md                    # Repository README
├── module-0/                    # Module 0 implementations
├── module-1/                    # Module 1 implementations
├── module-2/                    # Module 2 implementations
├── module-3/                    # Module 3 implementations
├── module-4/                    # Module 4 implementations
├── module-5/                    # Module 5 implementations
├── module-6/                    # Module 6 implementations
└── experiments/                 # Experimental graphs and tests
```

---

## 🔧 Git Configuration

### Your GitHub Credentials
- **Username**: Klement G
- **Email**: klementgunndu.singularity@gmail.com
- **GitHub Account**: klementgunndu

These are already configured globally on your system.

---

## 🚀 Automation Scripts

### 1. **Full Workflow Script** (`push_to_github.sh`)

This script handles:
- Git add
- Git commit with custom message
- Git push to GitHub

**Usage**:
```bash
./push_to_github.sh "Your commit message here"
```

**Example**:
```bash
./push_to_github.sh "Add Module 1 simple graph implementation"
```

### 2. **Quick Push Script** (`quick_push.sh`)

Auto-generates commit message based on changed files.

**Usage**:
```bash
./quick_push.sh
```

---

## 📝 Workflow for Claude Code

### When Claude Creates a File:

1. **Claude will create the Python file** in the appropriate module folder:
   ```
   /home/intruder/langgraph-git/module-1/simple_graph.py
   ```

2. **Claude will automatically run**:
   ```bash
   cd /home/intruder/langgraph-git
   ./push_to_github.sh "Add module-1: simple graph implementation"
   ```

3. **Result**: File is committed and pushed to GitHub

---

## 🎯 Usage Patterns

### Pattern 1: Single File Creation
```bash
# Claude creates file
/home/intruder/langgraph-git/module-1/router.py

# Claude runs
./push_to_github.sh "Add module-1: router with conditional edges"
```

### Pattern 2: Multiple Files
```bash
# Claude creates multiple files
module-2/state_schema.py
module-2/reducers.py

# Claude runs
./push_to_github.sh "Add module-2: state management examples"
```

### Pattern 3: Experiments
```bash
# Claude creates experimental file
experiments/custom_agent.py

# Claude runs
./push_to_github.sh "Experiment: custom agent with tool calling"
```

---

## 🔄 Git Commands Reference

### Manual Commands (if needed)

```bash
# Check status
git status

# Add specific file
git add module-1/simple_graph.py

# Add all changes
git add .

# Commit
git commit -m "Your message"

# Push
git push origin main

# Check remote
git remote -v

# View commit history
git log --oneline -10
```

---

## 📋 Commit Message Conventions

### Format:
```
Add <module>: <description>
```

### Examples:
```
Add module-1: simple graph with 3 nodes
Add module-2: state schema with Pydantic
Add module-3: breakpoint implementation
Add module-4: parallel execution example
Add module-5: memory store chatbot
Add module-6: production deployment config
Update module-1: improve error handling
Fix module-2: state reducer bug
Experiment: custom tool implementation
```

---

## 🌐 GitHub Repository Setup

### Initial Setup (First Time)

```bash
# 1. Create repository on GitHub
#    Repository name: langgraph-learning
#    Description: LangGraph Academy - Complete Implementation
#    Public or Private: Your choice

# 2. Link local repo to GitHub
git remote add origin git@github.com:klementgunndu/langgraph-learning.git

# 3. Push to GitHub
git push -u origin main
```

### Verify Setup

```bash
# Check remote URL
git remote -v

# Expected output:
# origin  git@github.com:klementgunndu/langgraph-learning.git (fetch)
# origin  git@github.com:klementgunndu/langgraph-learning.git (push)
```

---

## 🎨 Claude's Workflow Steps

### When you say: "Create simple_graph.py in langgraph-git"

**Claude will execute**:

1. **Create the file**:
   ```python
   # File: /home/intruder/langgraph-git/module-1/simple_graph.py
   ```

2. **Run automation script**:
   ```bash
   cd /home/intruder/langgraph-git
   ./push_to_github.sh "Add module-1: simple graph implementation"
   ```

3. **Confirm**:
   ```
   ✅ File created: module-1/simple_graph.py
   ✅ Committed to git
   ✅ Pushed to GitHub
   ```

---

## 🔍 Monitoring & Verification

### Check Git Status
```bash
cd /home/intruder/langgraph-git
git status
```

### View Recent Commits
```bash
git log --oneline -5
```

### View Remote Repository
Visit: `https://github.com/klementgunndu/langgraph-learning`

---

## 🛠️ Troubleshooting

### Issue: Permission Denied (SSH)
**Solution**: Make sure SSH key is added to GitHub account
```bash
cat ~/.ssh/id_rsa.pub
# Copy and add to GitHub: Settings → SSH Keys
```

### Issue: Remote Not Set
**Solution**:
```bash
git remote add origin git@github.com:klementgunndu/langgraph-learning.git
```

### Issue: Branch Mismatch
**Solution**:
```bash
git branch -M main
git push -u origin main
```

---

## 📚 Quick Reference

| Command | Purpose |
|---------|---------|
| `./push_to_github.sh "message"` | Full workflow: add, commit, push |
| `./quick_push.sh` | Auto-commit with generated message |
| `git status` | Check repository status |
| `git log --oneline -5` | View recent commits |
| `git remote -v` | Check remote URLs |

---

## ✅ Best Practices

1. **Clear commit messages**: Describe what was added/changed
2. **Module-based organization**: Keep files in module folders
3. **Regular commits**: Commit after completing each implementation
4. **Test before push**: Ensure code runs without errors
5. **Use automation**: Let scripts handle repetitive tasks

---

## 🎯 Your Learning Journey Tracking

This repository will contain:
- ✅ All 7 modules implementations
- ✅ Experimental graphs and agents
- ✅ Progress tracking via commits
- ✅ Portfolio of LangGraph projects

**GitHub URL**: `https://github.com/klementgunndu/langgraph-learning`

---

**Last Updated**: 2026-01-06
**Maintained by**: Claude Code + Klement G
