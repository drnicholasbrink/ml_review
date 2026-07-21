# AI Screening Guide - OpenAI Structured Outputs

## 🎯 Overview

This guide explains the updated AI-powered systematic review screening workflow using **OpenAI's latest API** with structured outputs.

## ✨ What Changed

### Old Approach (Pre-1.0 SDK)
- ❌ Used `openai<1.0.0` with deprecated API
- ❌ JSON parsing from text responses (unreliable)
- ❌ No schema enforcement
- ❌ Required manual validation

### New Approach (Latest SDK)
- ✅ Uses `openai>=1.0.0` with modern API
- ✅ **Native Pydantic support** for structured outputs
- ✅ **Schema strictly enforced** by OpenAI
- ✅ Type-safe, validated responses automatically
- ✅ Latest models: `gpt-4o`, `gpt-4o-mini`

## 🚀 Getting Started

### 1. Upgrade Dependencies

```bash
pip install --upgrade openai pydantic
```

Or update your conda environment:
```bash
conda env update -f environment.yml --prune
```

### 2. Verify Installation

Run the first cell in the notebook to check your OpenAI SDK version:
```python
import openai
print(openai.__version__)  # Should be >= 1.0.0
```

### 3. Run the Notebook

Follow the cells in order:
1. **Cell 0-1**: Install/verify dependencies
2. **Cell 2**: Setup imports and initialize OpenAI client
3. **Cell 3**: Define Pydantic schema for structured outputs
4. **Cell 4-5**: Load data and criteria
5. **Cell 6-7**: Understand screening function and test single abstract
6. **Cell 8-9**: Test on 10 abstracts
7. **Cell 10**: Run full screening (when ready)

## 📊 Model Selection

| Model | Use Case | Speed | Cost/1k abstracts | Quality |
|-------|----------|-------|-------------------|---------|
| **gpt-4o-mini** | Development, testing, large-scale | ⚡⚡⚡ | $2-5 | ⭐⭐⭐ |
| **gpt-4o** | Production screening | ⚡⚡ | $10-20 | ⭐⭐⭐⭐⭐ |
| **gpt-4-turbo** | Legacy, special cases | ⚡ | $40-80 | ⭐⭐⭐⭐ |

**Recommendation:**
- Start with `gpt-4o-mini` for development and testing
- Switch to `gpt-4o` for production screening
- Use dual screening (run twice) for critical decisions

## 🔧 Key Features

### Structured Outputs with Pydantic

```python
class ScreeningDecision(BaseModel):
    decision: Literal["include", "exclude", "uncertain"]
    confidence: Literal["high", "medium", "low"]
    exclusion_reason: Optional[Literal[...]]
    reasoning: str
    population_match: bool
    exposure_match: bool
    outcome_match: bool
    study_design_appropriate: bool
```

### Benefits:
- **Guaranteed valid JSON** - OpenAI enforces the schema
- **Type safety** - Pydantic validates all fields
- **No parsing errors** - Direct Pydantic object access
- **Consistent output** - Same structure every time

### API Call Example

```python
completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[...],
    response_format=ScreeningDecision,  # Pydantic model
    temperature=0.1
)

# Get parsed Pydantic object
result = completion.choices[0].message.parsed
```

## 📈 Workflow

### Development Phase
1. Test on 10 abstracts with `gpt-4o-mini`
2. Review decisions manually
3. Adjust prompts if needed
4. Calculate initial accuracy

### Validation Phase
1. Run on 100-200 abstracts
2. Compare with manual screening
3. Calculate sensitivity/specificity
4. Tune confidence thresholds

### Production Phase
1. Switch to `gpt-4o` for better accuracy
2. Run full screening with progress saving
3. Review all "uncertain" cases manually
4. Spot-check high/low confidence decisions

## 💰 Cost Estimation

### Example: 5,000 abstracts

**Development (gpt-4o-mini):**
- 10 abstracts test: ~$0.02
- 100 abstracts validation: ~$0.20
- Total development: ~$0.25

**Production (gpt-4o):**
- 5,000 abstracts: ~$50-100
- Re-screening uncertain (500): ~$5-10
- **Total: ~$55-110**

### Cost-Saving Tips:
1. Use `gpt-4o-mini` for initial screening
2. Use `gpt-4o` only for uncertain cases
3. Cache results - don't re-screen same abstracts
4. Use batch processing to minimize API calls

## ⚙️ Configuration

### Rate Limiting
```python
batch_screen_abstracts(
    df=df_pubmed,
    rate_limit_delay=0.5,  # seconds between calls
    batch_size=50  # save every 50 records
)
```

### Model Selection
```python
screen_abstract_with_ai(
    title=title,
    abstract=abstract,
    criteria=criteria,
    model="gpt-4o-mini"  # or "gpt-4o", "gpt-4-turbo"
)
```

## 📝 Output Format

Each screened abstract returns:

```json
{
  "decision": "include",
  "confidence": "high",
  "exclusion_reason": null,
  "reasoning": "Study examines heat exposure during pregnancy...",
  "population_match": true,
  "exposure_match": true,
  "outcome_match": true,
  "study_design_appropriate": true
}
```

## 🔍 Quality Assurance

### Validation Metrics
- **Sensitivity**: % of relevant studies identified
- **Specificity**: % of irrelevant studies excluded
- **Precision**: % of included studies that are relevant
- **F1 Score**: Balance of precision and recall

### Best Practices
1. **Double screening**: Run 10% of abstracts through manual review
2. **Inter-rater reliability**: Calculate kappa with human reviewers
3. **Uncertainty review**: Manually review all "uncertain" decisions
4. **Low confidence**: Review all decisions with "low" confidence
5. **Documentation**: Log all decisions and validation metrics

## 🛠️ Troubleshooting

### Issue: OpenAI SDK version error
**Solution:** Run `pip install --upgrade openai`

### Issue: Pydantic validation errors
**Solution:** Check that all enum values match schema exactly

### Issue: Rate limit errors
**Solution:** Increase `rate_limit_delay` parameter

### Issue: API timeouts
**Solution:** Implement exponential backoff (already built-in)

### Issue: Unexpected responses
**Solution:** Check system prompt, add few-shot examples

## 📚 Additional Resources

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [PRISMA Guidelines](http://www.prisma-statement.org/)

## 🎓 Example Workflow

```python
# 1. Load your data
df_pubmed = load_pubmed_data("pubmed_results_complete.csv")
criteria = load_inclusion_criteria("inclusion_criteria.txt")

# 2. Test on single abstract
result = screen_abstract_with_ai(
    title=df_pubmed.iloc[0]['Title'],
    abstract=df_pubmed.iloc[0]['Abstract'],
    criteria=criteria,
    model="gpt-4o-mini"
)

# 3. Run batch screening
results = batch_screen_abstracts(
    df=df_pubmed.head(100),  # Start small
    criteria=criteria,
    output_file="outputs/screening_results.csv",
    model="gpt-4o-mini"
)

# 4. Analyze results
analyze_screening_results("outputs/screening_results.csv")

# 5. Export for review
export_for_review("outputs/screening_results.csv")
```

## ✅ Summary

The updated notebook provides:
- ✅ Modern OpenAI API with structured outputs
- ✅ Type-safe Pydantic models
- ✅ Latest GPT-4o and GPT-4o-mini models
- ✅ Guaranteed valid, consistent responses
- ✅ Improved reliability and accuracy
- ✅ Better cost efficiency with model selection
- ✅ Production-ready workflow

**Ready to start screening!** 🚀
