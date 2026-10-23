"""Generates a research.summary for a faculty package using the OpenAI API.

Called automatically by ingest.py when research.summary is empty or missing.
Requires OPENAI_API_KEY in the environment (load from .env before running ingest).
"""

import os


def _build_prompt(pkg):
    identity = pkg.get("identity") or {}
    name = identity.get("canonical_name", "this faculty member")

    research = pkg.get("research") or {}
    domains = research.get("domains") or []
    methods = research.get("methods") or []
    applications = research.get("applications") or []

    pub_titles = [
        p["title"]
        for p in ((pkg.get("publications") or {}).get("records") or [])
        if p.get("title")
    ][:8]

    award_titles = [
        a["title"]
        for a in ((pkg.get("awards") or {}).get("records") or [])
        if a.get("title")
    ][:5]

    honor_texts = [
        h["honor"]
        for h in (pkg.get("honors") or [])
        if h.get("honor")
    ][:5]

    lines = [f"Write a concise, factual research summary paragraph (3–5 sentences) for {name}."]
    lines.append("Use only the information provided below. Do not invent details. Write in third person, present tense.")
    lines.append("")

    if domains:
        lines.append(f"Research domains: {', '.join(domains)}")
    if methods:
        lines.append(f"Methods: {', '.join(methods)}")
    if applications:
        lines.append(f"Applications: {', '.join(applications)}")
    if pub_titles:
        lines.append(f"Selected publications: {'; '.join(pub_titles)}")
    if award_titles:
        lines.append(f"Funded projects: {'; '.join(award_titles)}")
    if honor_texts:
        lines.append(f"Honors: {', '.join(honor_texts)}")

    return "\n".join(lines)


def generate_summary(pkg):
    """Returns a summary string, or None if the API key is missing."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        return None

    try:
        from openai import OpenAI
    except ImportError:
        print("  [summarize] openai package not installed — skipping summary generation")
        return None

    client = OpenAI(api_key=api_key)
    prompt = _build_prompt(pkg)

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [summarize] API error: {e}")
        return None
