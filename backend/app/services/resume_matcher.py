import json
from typing import Dict, List

import os

from app.services.ollama_client import ask_ollama
from app.services.openrouter_client import ask_openrouter


def ask_ai(prompt: str) -> str:
    provider = os.getenv("AI_PROVIDER", "ollama").lower()

    if provider == "openrouter":
        return ask_openrouter(prompt)

    return ask_ollama(prompt)

def analyze_resume_match(
    resume_text: str,
    job_description: str,
) -> Dict[str, object]:
    prompt = f"""
You are evaluating how well a candidate's resume matches a job posting.

Return ONLY valid JSON with exactly these keys:

{{
  "match_score": 0,
  "matched_skills": [],
  "missing_skills": [],
  "strengths": [],
  "gaps": [],
  "recommendations": []
}}

Rules:
- match_score must be the most accurate whole-number score from 0 to 100.
- Do not round match_score to multiples of 5 or 10.
- Scores such as 63, 71, 78, 84, or 92 are valid when justified by the actual resume-to-job match.
- Base match_score on the actual degree of alignment between the resume and job requirements.
- matched_skills must contain skills supported by BOTH the resume and job posting.
- missing_skills must contain important job requirements not supported by the resume.
- strengths must describe evidence-based advantages from the resume.
- gaps must describe meaningful weaknesses relative to the job.
- recommendations must give specific improvements to the resume or application.
- Do not invent skills, education, projects, or experience.
- Judge only from the supplied resume and job posting.
- Keep each item concise.
- Do not invent skills, education, projects, or experience.
- Judge only from the supplied resume and job posting.
- Keep each item concise.
- Return at most 6 matched_skills.
- Return at most 6 missing_skills.
- Return at most 4 strengths.
- Return at most 4 gaps.
- Return at most 4 recommendations.
- Each item must be one concise sentence.

RESUME:
{resume_text}

JOB POSTING:
{job_description}

RESUME:
{resume_text}

JOB POSTING:
{job_description}
"""

    raw_response = ask_ai(prompt)

try:
    result = json.loads(raw_response)
except (json.JSONDecodeError, TypeError):
    # AI occasionally returns an empty or malformed response.
    # Retry once automatically before failing.
    raw_response = ask_ai(prompt)

    try:
        result = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        raise ValueError("AI returned an invalid response after retrying.")

    score = result.get("match_score", 0)

    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 0

    result["match_score"] = max(0, min(100, score))

    missing_skills = result.get("missing_skills", [])

    if missing_skills:
        recommendations = []

        for skill in missing_skills[:5]:
            skill_lower = skill.lower()

            if any(word in skill_lower for word in ["tool", "software", "cad", "uvm"]):
                recommendation = (
                    f"Explore {skill} through a focused lab, tutorial, "
                    f"or small project that gives you hands-on exposure."
                )

            elif any(word in skill_lower for word in ["design", "pcb", "rtl", "vlsi"]):
                recommendation = (
                    f"Develop practical familiarity with {skill} by applying "
                    f"the underlying concepts in a small design project."
                )

            elif any(word in skill_lower for word in ["analysis", "integrity", "power", "noise"]):
                recommendation = (
                    f"Strengthen your understanding of {skill} through "
                    f"coursework, technical exercises, and hands-on analysis."
                )

            elif any(word in skill_lower for word in ["verification", "testing", "testbench"]):
                recommendation = (
                    f"Build experience with {skill} through verification labs, "
                    f"test exercises, or a relevant personal project."
                )

            else:
                recommendation = (
                    f"Build familiarity with {skill} through coursework, "
                    f"projects, labs, or hands-on practice."
                )

            recommendations.append(recommendation)

        result["recommendations"] = recommendations


    return result

def tailor_resume(resume_text: str, job_description: str):
    prompt = f"""
You are an expert resume editor and tailoring assistant.

Your job is to improve the candidate's resume for the target job while preserving factual accuracy.

CORE GOAL:
Make the resume wording stronger, clearer, more concise, and more relevant to the job.

CRITICAL TRUTHFULNESS RULES:

1. NEVER invent skills, tools, technologies, projects, coursework, responsibilities, metrics, achievements, methodologies, or experience.

2. NEVER claim the candidate used a technology unless that technology is explicitly supported somewhere in the resume.

3. Missing job requirements must remain missing. Put them in "missing_skills" rather than adding them to a suggested bullet, summary, keyword list, or skills-to-emphasize list.

4. Do not convert a missing skill into claimed experience.

5. Preserve the factual meaning of every bullet.

6. You may reuse technical context from elsewhere in the resume only when it clearly belongs to the same project, role, or experience.

7. Do not transfer a skill or technology from one unrelated project or role into another bullet.

8. Do not invent metrics, percentages, quantities, performance improvements, scale, impact, or business outcomes.

9. Do not imply professional or industry experience when the resume only describes coursework, academic projects, or personal projects.

10. Do not upgrade the candidate's proficiency level. For example, do not turn "introductory", "exposure", "basic", or coursework knowledge into "proficient", "expert", "advanced", or equivalent wording.

REWRITE QUALITY RULES:

11. Actively improve bullets whenever a meaningful wording improvement is possible.

12. Do NOT simply copy the original bullet unless it is genuinely already strong and cannot be improved without changing its meaning.

13. Suggested bullets should normally be noticeably stronger than the originals while remaining truthful.

14. Prefer strong, specific action verbs such as:
- designed
- developed
- implemented
- validated
- analyzed
- integrated
- tested
- created
- configured

Use them only when factually equivalent to what the resume says.

15. Remove weak or unnecessary wording such as:
- "simple"
- "helped"
- "worked on"
- "responsible for"
- "gained experience"
- "learned about"
- unnecessary filler

16. Improve conciseness, clarity, readability, technical specificity, and professional tone.

17. Reorder wording when doing so makes the technical contribution easier to understand.

18. Prefer accomplishment/action-oriented wording over learning-oriented wording.

Avoid phrases such as:
- "enhanced proficiency"
- "strengthened understanding"
- "gained knowledge"
- "improved understanding"

Instead, describe the concrete work the candidate performed when the resume supports it.

19. Do not make a bullet longer merely to make it sound more impressive. Prefer concise, information-dense wording.

20. Do not add generic filler such as:
- "with a focus on"
- "with emphasis on"
- "to ensure optimal performance"
- "to improve efficiency"
unless the resume explicitly supports that purpose or outcome.

TECHNICAL CONTEXT RULES:

21. You may add a technology to a rewritten bullet when:
- the technology is explicitly present elsewhere in the resume, AND
- it clearly belongs to the same project or experience.

22. You may add project context, such as "4-bit CPU", when that context is explicitly established elsewhere in the same project.

23. Do NOT add implementation goals, testing methodologies, verification strategies, debugging methods, architectural decisions, or responsibilities unless they are explicitly supported by the resume.

24. Do not infer plausible engineering work. Something being likely does not mean the candidate did it.

25. Do not add words such as "randomized testing", "directed testing", "coverage-driven", "optimized", "automated", "production", "scalable", "high-performance", or similar technical claims unless explicitly supported.

TAILORING RULES:

26. Tailor wording toward the job description using ONLY facts supported by the resume.

27. Skills already supported by the resume may be emphasized when relevant to the job.

28. "skills_to_emphasize" must contain ONLY skills actually supported by the resume.

29. "keywords_to_add" must contain ONLY terminology that is already supported by the candidate's resume but could be stated more clearly.

30. Never place an unsupported job-description keyword in "keywords_to_add".

31. Unsupported job requirements belong in "missing_skills".

32. General recommendations may suggest learning or gaining experience with missing skills, but must not tell the candidate to simply add those skills to the resume.

33. If the candidate may already have relevant experience but it is not documented in the resume, say something like:
"If you have experience with X, consider documenting it."
Do not assume they have it.

SUMMARY RULES:

34. The suggested summary must not introduce any skill, experience, specialization, or proficiency that is unsupported by the resume.

35. The suggested summary should be concise and targeted to the job while preserving the candidate's actual background.

36. Do not change education level, job title, degree, seniority, or years of experience.

BULLET REWRITE RULES:

37. Every "original" value must reproduce the actual original bullet being rewritten.

38. Every "suggested" value must be a polished replacement for that specific bullet.

39. Every "reason" must explain the meaningful improvement, such as:
- stronger action verb
- improved conciseness
- clearer technical context
- improved relevance
- removal of weak wording
- improved readability

40. Do not say "no change needed" unless there is genuinely no worthwhile improvement.

41. If a bullet is already strong, small improvements in conciseness or wording are acceptable, but do not force unnecessary changes.

42. Do not combine facts from multiple unrelated bullets into one rewrite.

43. Do not remove important technical details merely to make a bullet shorter.

44. Preserve technologies, quantities, hardware specifications, programming languages, and other important factual details from the original when relevant.

45. The final suggested bullet should read naturally as a resume bullet, not as an explanation or paragraph.

GOOD EXAMPLE:

Original:
"Built a testbench to validate instruction execution and control flow."

Improved:
"Developed a Verilog testbench to validate CPU instruction execution and control-flow behavior."

This rewrite is allowed ONLY if Verilog and the CPU project are supported elsewhere in the resume.

Another example:

Original:
"Designed and tested digital circuits and state machines in Verilog and MATLAB."

Improved:
"Designed and validated digital circuits and finite-state machines using Verilog and MATLAB."

This improves wording without inventing experience.

BAD EXAMPLE:

Original:
"Implemented register file, ALU, and instruction decoding logic."

Bad suggested:
"Implemented register file, ALU, and instruction decoding logic."

Do not simply repeat the original unless there is genuinely no meaningful improvement possible.

If a bullet genuinely should not change, it may remain unchanged, but this should be uncommon.

Return ONLY valid JSON in exactly this structure:

{{
  "summary_suggestion": "string",
  "keywords_to_add": ["string"],
  "bullet_rewrites": [
    {{
      "original": "string",
      "suggested": "string",
      "reason": "string"
    }}
  ],
  "skills_to_emphasize": ["string"],
  "missing_skills": ["string"],
  "general_recommendations": ["string"]
}}

For each bullet rewrite:
- "original" must contain the original bullet.
- "suggested" must contain an improved version.
- "reason" must briefly explain what was improved, such as stronger verb, conciseness, clarity, relevance, or technical specificity.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}
"""

    raw_response = ask_ai(prompt)
    result = json.loads(raw_response)

    resume_lower = resume_text.lower()

    result["skills_to_emphasize"] = [
        skill
        for skill in result.get("skills_to_emphasize", [])
        if skill.lower() in resume_lower
    ]

    result["keywords_to_add"] = [
        keyword
        for keyword in result.get("keywords_to_add", [])
        if keyword.lower() in resume_lower
    ]

    safe_rewrites = []

    blocked_terms = [
        "signal integrity",
        "formal verification",
        "uvm",
        "coverage-driven",
        "coverage driven",
        "post-silicon",
        "post silicfor rewriteon",
        "power integrity",
        "randomized testing",
    ]

    experience_upgrades = [
        "gained practical experience",
        "gained hands-on experience",
        "gained hands on experience",
        "proficient",
        "expert",
        "advanced experience",
    ]

    for rewrite in result.get("bullet_rewrites", []):
        # Ignore malformed model output instead of letting it crash
        # the entire tailoring response.
        if not isinstance(rewrite, dict):
            continue

        original = str(rewrite.get("original") or "").strip()
        suggested = str(rewrite.get("suggested") or "").strip()
        reason = str(rewrite.get("reason") or "").strip()

        # A rewrite is useless if we cannot identify the original bullet.
        if not original:
            continue

        # Guarantee every rewrite satisfies the Pydantic schema.
        if not suggested:
            suggested = original

        if not reason:
            reason = "No safe wording improvement was provided."

        original_lower = original.lower()
        suggested_lower = suggested.lower()

        introduced_blocked_term = any(
            term in suggested_lower and term not in original_lower
            for term in blocked_terms
        )

        introduced_experience_upgrade = any(
            phrase in suggested_lower and phrase not in original_lower
            for phrase in experience_upgrades
        )

        if introduced_blocked_term or introduced_experience_upgrade:
            safe_rewrites.append(
                {
                    "original": original,
                    "suggested": original,
                    "reason": (
                        "Rejected because the rewrite introduced an "
                        "unsupported or stronger claim."
                    ),
                }
            )
        else:
            safe_rewrites.append(
                {
                    "original": original,
                    "suggested": suggested,
                    "reason": reason,
                }
            )

    result["bullet_rewrites"] = safe_rewrites
    return result
    
