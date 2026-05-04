# Fundamentals of AI Engineering - 6-Week Beginner Program

## Course Overview

The **Fundamentals of AI Engineering** course is designed for AI beginners, including learners with non-technical backgrounds. It follows a progressive model:

**Agent Tools Introduction -> AI-Assisted Code Practice -> Technical Practice -> Capstone Delivery**

The course emphasizes practical confidence, conceptual intuition, AI-assisted learning, and reproducible project delivery. Students first learn how to collaborate with AI tools, then gradually use those tools to understand code, process data, call LLMs, run basic ML experiments, and complete a small end-to-end AI project.

## Target Students

- **Complete Beginners**: Learners with no programming experience who want to enter AI, data analytics, or AI-assisted development.
- **Tool Users**: Professionals already using AI tools who want stronger technical foundations.
- **Technical Learners**: Developers or analysts who want a beginner-friendly path through core AI engineering habits.

## Prerequisites

### Weeks 1-2: No Programming Required

- Ability to use a browser and create online accounts.
- Ability to install and open desktop software.
- Basic computer operation skills.

### Weeks 3-6: Introductory Programming With AI Assistance

- Basic terminal operations will be introduced in class.
- Python concepts are taught through guided examples and AI-assisted practice.
- Self-learn materials are optional refreshers, not required prerequisites.

## Learn AI With AI

The core philosophy is **"Learn AI with AI"**:

1. **Weeks 1-2**: Learn to use AI tools such as ChatGPT, Claude, Cursor, Kilo, and GitHub Copilot Chat.
2. **Weeks 3-6**: Use AI tools to help read code, debug errors, explain concepts, generate first drafts, and validate results.

AI is a learning accelerator, not a replacement for understanding. Students are expected to explain what they submitted, document how AI helped, and verify outputs before trusting them.

## Course Navigation

- [Pre-Study Guide](PRESTUDY.md) - Optional preparation before the course.
- [Course Syllabus](SYLLABUS.md) - Detailed 6-week schedule.
- [Assignment Requirements](assignments.md) - Weekly assessment criteria.
- [Capstone Project](capstone.md) - Final intelligent data analysis script.

### Optional Self-Study References

Use these only when you want extra background or review:

- [Self-Study Schedule](self_learn/Schedule.md)
- [Chapter 1: Tool Preparation](self_learn/Chapters/1/Chapter1.md)
- [Chapter 2: Python and Environment Management](self_learn/Chapters/2/Chapter2.md)
- [Chapter 3: AI Engineering Basics](self_learn/Chapters/3/Chapter3.md)
- [Chapter 4: Hugging Face Platform and Local Inference](self_learn/Chapters/4/Chapter4.md)
- [Chapter 5: Resource Monitoring and Containerization](self_learn/Chapters/5/Chapter5.md)

The [old_v1/](old_v1/) directory contains the previous 8-week course and is kept as an archive/reference for deeper self-study.

## Course Duration

- **Total Duration**: 6 weeks
- **Weekly Hours**: 5 hours recommended
- **Suggested Rhythm**:
  - Session 1: Core concepts and examples
  - Session 2: Hands-on guided practice
  - Session 3: Q&A, review, and project support

## 6-Week Teaching Plan

| Week | Theme | Main Deliverable |
|------|-------|------------------|
| Week 1 | Agent Tools Introduction | AI tool experience report, representative prompts, and reflection |
| Week 2 | IDE + AI-Assisted Code Practice | Code reading notes, 2-3 small modifications, and one debugging record |
| Week 3 | Environment + Data Processing | CSV data profiling script with `profile.json` and `profile.md` |
| Week 4 | LLM Fundamentals + API Reliability | Structured-output demo and simplified reliable LLM client notes |
| Week 5 | ML Training Loop | Two lightweight baseline runs and a short comparison report |
| Week 6 | Capstone: Intelligent Data Analysis Script | CSV -> `report.json` + `report.md` + short demo |

## Learning Outcomes

By the end of the course, students will be able to:

1. **Use AI agent tools effectively**
   - Select appropriate tools for writing, research, code explanation, and debugging.
   - Document useful prompts and reflect on tool strengths and limits.

2. **Read, modify, and debug code with AI assistance**
   - Explain simple Python functions.
   - Make small changes safely.
   - Use error messages and AI feedback to debug.

3. **Process data reproducibly**
   - Create a clean Python environment.
   - Load and inspect CSV data with pandas.
   - Generate stable JSON and Markdown profiling reports.

4. **Build basic LLM workflows**
   - Explain tokens, context windows, and structured prompts.
   - Request JSON-like outputs and validate them.
   - Add beginner-friendly timeout, retry, and logging practices.

5. **Understand the ML training loop**
   - Train a basic baseline model.
   - Save metrics and compare two runs.
   - Explain results without relying on advanced math.

6. **Deliver a small end-to-end AI project**
   - Build a reproducible intelligent data analysis script.
   - Combine data profiling, LLM explanation, and report generation.
   - Demonstrate the project and explain key decisions.

## Assessment Method

| Component | Weight |
|-----------|--------|
| Week 1: Agent Tools Experience Report | 10% |
| Week 2: AI-Assisted Code Practice | 10% |
| Week 3: Data Profiling Report | 15% |
| Week 4: LLM Structured Output + Reliability Practice | 15% |
| Week 5: ML Baseline Comparison | 10% |
| Week 6: Intelligent Data Analysis Capstone | 25% |
| Participation | 15% |
| **Total** | **100%** |

## Technology Stack

### Weeks 1-2: Tool Usage Phase

- **AI Chat Tools**: ChatGPT / Claude
- **AI Editor**: Cursor or VS Code with AI assistant
- **AI Programming Assistants**: Kilo, GitHub Copilot Chat, or equivalents

### Weeks 3-6: Technical Practice Phase

- **Language**: Python 3.10+ (3.11 recommended)
- **Core Libraries**: `pandas`, `numpy`, `scikit-learn`
- **Engineering Tools**: virtual environments, `requirements.txt`, basic logging, optional `pytest`
- **LLM Options**: hosted API or local inference; Week 4 requires only one working path

## Weekly Resources

- [Week 1](week_01/README.md): Agent tools introduction
- [Week 2](week_02/README.md): IDE and AI-assisted code practice
- [Week 3](week_03/README.md): Environment and data processing
- [Week 4](week_04/README.md): LLM fundamentals and API reliability
- [Week 5](week_05/README.md): ML training loop
- [Week 6](week_06/README.md): Capstone project

## Learning Path After This Course

After finishing this beginner program, students can continue with:

- RAG fundamentals and vector databases
- Agent systems and tool use
- Production deployment with FastAPI or Docker
- Advanced local inference and model evaluation

## How to Ask for Help

When asking instructors, classmates, or AI tools for help, include:

1. Goal: what you are trying to do.
2. Context: which week and which step.
3. Exact command and full output.
4. Environment: OS, Python version, dependency file.
5. What you already tried.

This course values evidence-based debugging: reproduce the issue, capture the details, and then ask for help.

---

**Course Version**: v2.1 (6-Week Beginner Program)
**Last Updated**: 2026-05-04
**Major Changes**: Week 3 now focuses on data processing; Week 4 now covers LLM fundamentals and simplified API reliability; Week 6 is unified around the intelligent data analysis capstone.
