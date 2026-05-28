"""
Real-agent case study: cross-component arbitrage in a deployed-style
multi-tool LLM ensemble.

Setup. We instantiate a planner-style multi-tool agent that delegates each
outcome of a multi-candidate partition forecasting question to a different
specialist LLM. Specialist a sees ONLY its assigned outcome (single-question
prompt; no context about the partition or the other outcomes). It returns
K=8 verbalized probability samples; the mean is its component marginal. The
agent assembles the per-outcome marginals into a joint quote and we measure
the compositional residual eps^star = ||p - Pi*(p)||_2 against the joint
partition polytope (sum=1, p>=0).

This is the "deployed multi-tool" failure mode promised in the paper:
each component is locally coherent on its single-Bernoulli question
(trivially: no constraint), but the assembled joint quote violates the
cross-component partition coupling.
"""

from __future__ import annotations
import os, sys, json, random, argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Wire JCD repo into path so we can import its LLM clients and projection.
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # JCD-Forecasting
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from jcd.eval.sample import (
    AnthropicClient, AzureOpenAIClient, GroqClient,
    parse_verbalized_probability, DEFAULT_PROMPT,
)
import logging
log = logging.getLogger(__name__)


# GPT-5.4 series Azure deployments require max_completion_tokens (not
# max_tokens). The shipped AzureOpenAIClient still uses max_tokens, so we
# subclass to swap the parameter name. Everything else is inherited.
from dataclasses import dataclass
@dataclass
class AzureGPT54Client(AzureOpenAIClient):
    def forecast_one(self, question, *, temperature=0.7, seed=None):
        prompt = self.prompt_template.format(
            title=question.title,
            body=question.body,
            resolution_date=question.resolution_date or "unspecified",
        )
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=64,
                )
                text = resp.choices[0].message.content or ""
                p = parse_verbalized_probability(text)
                if p is not None:
                    return p
            except Exception as e:
                log.warning("Azure (%s) request failed (attempt %d): %s",
                            self._deployment, attempt + 1, e)
        return None
from jcd.data.paleka import PalekaQuestion
from jcd.types import Clique, Relation
from jcd.qp.solver import project as jcd_project

K = 8
SEED = 0
TEMPERATURE = 0.7

# ---------------------------------------------------------------------------
# Partition forecasting questions. Each is (label, [outcome strings],
# resolution_date). The agent will route each outcome to one specialist.
# ---------------------------------------------------------------------------
PARTITIONS: list[dict] = [
    # ----- Politics (4) -----
    dict(label="2028 US Presidential winner", date="2028-11-07", outcomes=[
        "the Republican Party nominee wins the 2028 US Presidential election",
        "the Democratic Party nominee wins the 2028 US Presidential election",
        "an independent or third-party candidate wins the 2028 US Presidential election",
    ]),
    dict(label="2026 US House majority", date="2027-01-03", outcomes=[
        "the Republican Party holds an outright majority in the US House of Representatives following the 2026 midterm elections",
        "the Democratic Party holds an outright majority in the US House of Representatives following the 2026 midterm elections",
        "no party holds an outright majority in the US House of Representatives following the 2026 midterm elections",
    ]),
    dict(label="2026 US Senate majority", date="2027-01-03", outcomes=[
        "the Republican Party holds an outright majority in the US Senate following the 2026 midterm elections",
        "the Democratic Party holds an outright majority in the US Senate following the 2026 midterm elections",
        "no party holds an outright majority in the US Senate following the 2026 midterm elections",
    ]),
    dict(label="Next UK general election year", date="2029-12-31", outcomes=[
        "the next UK general election occurs in calendar year 2026",
        "the next UK general election occurs in calendar year 2027",
        "the next UK general election occurs in calendar year 2028",
        "the next UK general election occurs in 2029 or later",
    ]),
    # ----- Macro / finance (5) -----
    dict(label="Next FOMC decision (June 2026 meeting)", date="2026-06-18", outcomes=[
        "the US Federal Reserve cuts the federal funds rate by 25bps or more at the June 2026 FOMC meeting",
        "the US Federal Reserve holds the federal funds rate unchanged at the June 2026 FOMC meeting",
        "the US Federal Reserve raises the federal funds rate by 25bps or more at the June 2026 FOMC meeting",
    ]),
    dict(label="2026 H1 SP500 return bucket", date="2026-06-30", outcomes=[
        "the SP500 index closes 2026-H1 down more than 5% year-to-date",
        "the SP500 index closes 2026-H1 between -5% and +5% year-to-date",
        "the SP500 index closes 2026-H1 up between 5% and 15% year-to-date",
        "the SP500 index closes 2026-H1 up more than 15% year-to-date",
    ]),
    dict(label="US unemployment rate, June 2026", date="2026-07-05", outcomes=[
        "the US unemployment rate (BLS U-3) for June 2026 is below 4.0%",
        "the US unemployment rate (BLS U-3) for June 2026 is between 4.0% and 4.5%",
        "the US unemployment rate (BLS U-3) for June 2026 is between 4.5% and 5.0%",
        "the US unemployment rate (BLS U-3) for June 2026 is above 5.0%",
    ]),
    dict(label="2026 H1 BTC return bucket", date="2026-06-30", outcomes=[
        "Bitcoin closes 2026-H1 down more than 10% year-to-date",
        "Bitcoin closes 2026-H1 within +/-10% year-to-date",
        "Bitcoin closes 2026-H1 up more than 10% year-to-date",
    ]),
    dict(label="2026 US CPI year-over-year, December print", date="2027-01-15", outcomes=[
        "US CPI year-over-year for December 2026 is below 2.0%",
        "US CPI year-over-year for December 2026 is between 2.0% and 3.0%",
        "US CPI year-over-year for December 2026 is between 3.0% and 4.0%",
        "US CPI year-over-year for December 2026 is above 4.0%",
    ]),
    # ----- Sports (4) -----
    dict(label="Top gold-medal nation, 2028 Summer Olympics", date="2028-08-15", outcomes=[
        "the United States wins the most gold medals at the 2028 Summer Olympics in Los Angeles",
        "China wins the most gold medals at the 2028 Summer Olympics in Los Angeles",
        "a country other than the United States or China wins the most gold medals at the 2028 Summer Olympics",
    ]),
    dict(label="2026 FIFA World Cup winner confederation", date="2026-07-19", outcomes=[
        "the 2026 FIFA World Cup is won by a UEFA (European) national team",
        "the 2026 FIFA World Cup is won by a CONMEBOL (South American) national team",
        "the 2026 FIFA World Cup is won by a national team from any other confederation",
    ]),
    dict(label="2026 Wimbledon men's singles winner", date="2026-07-13", outcomes=[
        "Carlos Alcaraz wins the 2026 Wimbledon men's singles championship",
        "Jannik Sinner wins the 2026 Wimbledon men's singles championship",
        "Novak Djokovic wins the 2026 Wimbledon men's singles championship",
        "the 2026 Wimbledon men's singles championship is won by some other player",
    ]),
    dict(label="2026 NBA Finals winner conference", date="2026-06-30", outcomes=[
        "the 2026 NBA Finals is won by an Eastern Conference team",
        "the 2026 NBA Finals is won by a Western Conference team",
    ]),
    # ----- Awards / culture (4) -----
    dict(label="2026 Booker Prize winner nationality", date="2026-11-15", outcomes=[
        "the 2026 Booker Prize for Fiction is won by an author from the United Kingdom",
        "the 2026 Booker Prize for Fiction is won by an author from the United States",
        "the 2026 Booker Prize for Fiction is won by an author from a country other than the United Kingdom or the United States",
    ]),
    dict(label="2026 Academy Award Best Picture winner type", date="2027-03-08", outcomes=[
        "the 2026 Academy Award for Best Picture (presented in March 2027) goes to a major-studio production",
        "the 2026 Academy Award for Best Picture (presented in March 2027) goes to a streaming-platform production (e.g. Netflix, Apple TV, Amazon)",
        "the 2026 Academy Award for Best Picture (presented in March 2027) goes to an independent / arthouse production",
    ]),
    dict(label="2026 Grammy Album of the Year genre", date="2027-02-08", outcomes=[
        "the 2026 Grammy Award for Album of the Year is won by a primarily pop record",
        "the 2026 Grammy Award for Album of the Year is won by a primarily hip-hop or R&B record",
        "the 2026 Grammy Award for Album of the Year is won by a primarily rock or alternative record",
        "the 2026 Grammy Award for Album of the Year is won by a record in some other genre",
    ]),
    dict(label="2026 Pulitzer Prize Fiction winner setting", date="2026-05-04", outcomes=[
        "the 2026 Pulitzer Prize for Fiction is awarded to a novel set primarily in the United States",
        "the 2026 Pulitzer Prize for Fiction is awarded to a novel set primarily outside the United States",
        "the 2026 Pulitzer Prize for Fiction is not awarded in 2026",
    ]),
    # ----- Tech (5) -----
    dict(label="Top MMLU score, year-end 2026", date="2026-12-31", outcomes=[
        "by 31 December 2026, the highest publicly reported MMLU score is held by an Anthropic Claude model",
        "by 31 December 2026, the highest publicly reported MMLU score is held by an OpenAI GPT model",
        "by 31 December 2026, the highest publicly reported MMLU score is held by a Google or DeepMind Gemini model",
        "by 31 December 2026, the highest publicly reported MMLU score is held by a Meta Llama model or any other organisation",
    ]),
    dict(label="Top SWE-Bench Verified score, year-end 2026", date="2026-12-31", outcomes=[
        "by 31 December 2026, the highest publicly reported SWE-Bench Verified score is held by an Anthropic Claude model",
        "by 31 December 2026, the highest publicly reported SWE-Bench Verified score is held by an OpenAI GPT model",
        "by 31 December 2026, the highest publicly reported SWE-Bench Verified score is held by a Google Gemini model",
        "by 31 December 2026, the highest publicly reported SWE-Bench Verified score is held by some other model",
    ]),
    dict(label="Largest US tech IPO of 2026 by company type", date="2026-12-31", outcomes=[
        "the largest US tech IPO of 2026 (by initial market capitalisation) is an AI infrastructure or AI model company",
        "the largest US tech IPO of 2026 is a SaaS / enterprise software company",
        "the largest US tech IPO of 2026 is a consumer-internet or fintech company",
        "the largest US tech IPO of 2026 is in some other technology subsector",
    ]),
    dict(label="Next iPhone announcement quarter", date="2026-12-31", outcomes=[
        "the next major iPhone hardware announcement after May 2026 occurs in calendar Q3 2026 (Jul-Sep)",
        "the next major iPhone hardware announcement after May 2026 occurs in calendar Q4 2026 (Oct-Dec)",
        "the next major iPhone hardware announcement after May 2026 occurs in 2027 or later",
    ]),
    dict(label="Top semiconductor maker by revenue, 2026", date="2027-02-15", outcomes=[
        "TSMC is the world's largest semiconductor company by 2026 annual revenue",
        "Nvidia is the world's largest semiconductor company by 2026 annual revenue",
        "a different company is the world's largest semiconductor company by 2026 annual revenue",
    ]),
    # ----- Science / Nobel (4) -----
    dict(label="2026 Nobel Peace Prize laureate type", date="2026-12-10", outcomes=[
        "the 2026 Nobel Peace Prize is awarded to one or more individual persons (and not an organisation)",
        "the 2026 Nobel Peace Prize is awarded to an organisation (NGO, treaty body, agency, etc.)",
        "the 2026 Nobel Peace Prize is not awarded in 2026",
    ]),
    dict(label="2026 Nobel Chemistry laureate field", date="2026-12-10", outcomes=[
        "the 2026 Nobel Prize in Chemistry recognises work primarily in biochemistry or molecular biology",
        "the 2026 Nobel Prize in Chemistry recognises work primarily in materials, nanotechnology, or polymer chemistry",
        "the 2026 Nobel Prize in Chemistry recognises work primarily in catalysis or organic synthesis",
        "the 2026 Nobel Prize in Chemistry recognises work in some other chemistry subfield",
    ]),
    dict(label="2026 Nobel Physics laureate field", date="2026-12-10", outcomes=[
        "the 2026 Nobel Prize in Physics recognises work primarily in particle physics or cosmology",
        "the 2026 Nobel Prize in Physics recognises work primarily in condensed matter or materials physics",
        "the 2026 Nobel Prize in Physics recognises work primarily in quantum information or atomic/molecular optics",
        "the 2026 Nobel Prize in Physics recognises work in some other physics subfield",
    ]),
    dict(label="2026 ACM Turing Award area", date="2027-04-15", outcomes=[
        "the 2026 ACM Turing Award (announced in early 2027) recognises work in machine learning or artificial intelligence",
        "the 2026 ACM Turing Award recognises work in systems, programming languages, or theory",
        "the 2026 ACM Turing Award recognises work in some other area of computer science",
    ]),
    # ----- Demographic / governance (4) -----
    dict(label="2026 EU AI Act state of enforcement", date="2026-12-31", outcomes=[
        "the EU AI Act's general-purpose-AI obligations are fully in force and actively enforced by year-end 2026",
        "the EU AI Act's general-purpose-AI obligations are nominally in force but enforcement is delayed or limited at year-end 2026",
        "the EU AI Act's general-purpose-AI obligations are deferred or significantly modified beyond year-end 2026",
    ]),
    dict(label="UN climate ambition state, COP31 (2026)", date="2026-11-30", outcomes=[
        "the COP31 climate conference (2026) produces an enhanced global ambition / phase-out commitment relative to COP30",
        "the COP31 climate conference (2026) produces only incremental or text-level adjustments relative to COP30",
        "the COP31 climate conference (2026) ends without a substantive joint declaration",
    ]),
    dict(label="2026 net global solar capacity addition", date="2027-03-31", outcomes=[
        "global net solar PV capacity additions in calendar 2026 are below 500 GW",
        "global net solar PV capacity additions in calendar 2026 are between 500 and 700 GW",
        "global net solar PV capacity additions in calendar 2026 are above 700 GW",
    ]),
    dict(label="World population, mid-2027", date="2027-07-01", outcomes=[
        "world population on 1 July 2027 (UN World Population Prospects) is below 8.10 billion",
        "world population on 1 July 2027 is between 8.10 and 8.20 billion",
        "world population on 1 July 2027 is above 8.20 billion",
    ]),
    # ===== Extension to N=100: 70 additional partitions, all resolving
    # strictly after 1 May 2026 (the leakage cutoff).
    # ----- Politics (12) -----
    dict(label="2026 California gubernatorial party", date="2026-11-03", outcomes=[
        "the winner of the 2026 California gubernatorial election is a Democrat",
        "the winner of the 2026 California gubernatorial election is a Republican",
        "the winner of the 2026 California gubernatorial election is from a different party",
    ]),
    dict(label="2026 Texas gubernatorial party", date="2026-11-03", outcomes=[
        "the winner of the 2026 Texas gubernatorial election is a Republican",
        "the winner of the 2026 Texas gubernatorial election is a Democrat",
        "the winner of the 2026 Texas gubernatorial election is from a different party",
    ]),
    dict(label="2026 Florida gubernatorial party", date="2026-11-03", outcomes=[
        "the winner of the 2026 Florida gubernatorial election is a Republican",
        "the winner of the 2026 Florida gubernatorial election is a Democrat",
        "the winner of the 2026 Florida gubernatorial election is from a different party",
    ]),
    dict(label="2026 New York gubernatorial party", date="2026-11-03", outcomes=[
        "the winner of the 2026 New York gubernatorial election is a Democrat",
        "the winner of the 2026 New York gubernatorial election is a Republican",
        "the winner of the 2026 New York gubernatorial election is from a different party",
    ]),
    dict(label="2026 Ohio gubernatorial party", date="2026-11-03", outcomes=[
        "the winner of the 2026 Ohio gubernatorial election is a Republican",
        "the winner of the 2026 Ohio gubernatorial election is a Democrat",
        "the winner of the 2026 Ohio gubernatorial election is from a different party",
    ]),
    dict(label="2026 Pennsylvania US Senate election", date="2026-11-03", outcomes=[
        "the 2026 Pennsylvania US Senate election is won by the Democratic nominee",
        "the 2026 Pennsylvania US Senate election is won by the Republican nominee",
        "the 2026 Pennsylvania US Senate election is won by a candidate from a different party",
    ]),
    dict(label="2027 French Presidential winner ideology", date="2027-05-09", outcomes=[
        "the 2027 French Presidential election is won by a candidate from the political left",
        "the 2027 French Presidential election is won by a candidate from the political centre",
        "the 2027 French Presidential election is won by a candidate from the political right",
    ]),
    dict(label="2026 Brazilian general election composition", date="2026-10-31", outcomes=[
        "after the 2026 Brazilian general election, the largest bloc in the Chamber of Deputies is aligned with President Lula's coalition",
        "after the 2026 Brazilian general election, the largest bloc in the Chamber of Deputies is aligned with the Bolsonaro-aligned opposition",
        "after the 2026 Brazilian general election, the largest bloc in the Chamber of Deputies is a centrist or independent bloc",
    ]),
    dict(label="2026 German Bundestag composition (post-election)", date="2026-12-31", outcomes=[
        "after the September 2026 (or earlier) German federal election, the federal government is led by the CDU/CSU bloc",
        "after the German federal election, the federal government is led by the SPD",
        "after the German federal election, the federal government is led by another party or coalition lead",
    ]),
    dict(label="2027 UK Prime Minister tenure", date="2027-12-31", outcomes=[
        "Keir Starmer is still Prime Minister of the United Kingdom on 31 December 2027",
        "the United Kingdom has a different Labour Prime Minister on 31 December 2027",
        "the United Kingdom has a Conservative or other-party Prime Minister on 31 December 2027",
    ]),
    dict(label="2027 Mexico midterm legislative balance", date="2027-06-06", outcomes=[
        "after the 2027 Mexican midterm elections, the largest bloc in the Chamber of Deputies is aligned with Morena",
        "after the 2027 Mexican midterm elections, the largest bloc in the Chamber of Deputies is aligned with PAN",
        "after the 2027 Mexican midterm elections, the largest bloc in the Chamber of Deputies is from a different party",
    ]),
    dict(label="2026 Japan LDP leadership", date="2026-12-31", outcomes=[
        "the leader of Japan's Liberal Democratic Party on 31 December 2026 is the same person who led the LDP at the start of 2026",
        "the leader of Japan's Liberal Democratic Party on 31 December 2026 is a different LDP politician than at the start of 2026",
        "Japan does not have a Liberal Democratic Party leader on 31 December 2026 (party dissolution or merger)",
    ]),
    # ----- Macro / finance (12) -----
    dict(label="2026 H2 SP500 return bucket", date="2026-12-31", outcomes=[
        "the SP500 returns less than -5% over 2026-H2 (1 Jul to 31 Dec)",
        "the SP500 returns between -5% and +5% over 2026-H2",
        "the SP500 returns between +5% and +15% over 2026-H2",
        "the SP500 returns more than +15% over 2026-H2",
    ]),
    dict(label="2027 H1 SP500 return bucket", date="2027-06-30", outcomes=[
        "the SP500 returns less than -5% over 2027-H1",
        "the SP500 returns between -5% and +5% over 2027-H1",
        "the SP500 returns more than +5% over 2027-H1",
    ]),
    dict(label="December 2026 FOMC decision", date="2026-12-15", outcomes=[
        "the US Federal Reserve cuts the federal funds rate by 25bps or more at the December 2026 FOMC meeting",
        "the US Federal Reserve holds the federal funds rate unchanged at the December 2026 FOMC meeting",
        "the US Federal Reserve raises the federal funds rate by 25bps or more at the December 2026 FOMC meeting",
    ]),
    dict(label="March 2027 FOMC decision", date="2027-03-17", outcomes=[
        "the US Federal Reserve cuts the federal funds rate by 25bps or more at the March 2027 FOMC meeting",
        "the US Federal Reserve holds the federal funds rate unchanged at the March 2027 FOMC meeting",
        "the US Federal Reserve raises the federal funds rate by 25bps or more at the March 2027 FOMC meeting",
    ]),
    dict(label="June 2027 FOMC decision", date="2027-06-16", outcomes=[
        "the US Federal Reserve cuts the federal funds rate by 25bps or more at the June 2027 FOMC meeting",
        "the US Federal Reserve holds the federal funds rate unchanged at the June 2027 FOMC meeting",
        "the US Federal Reserve raises the federal funds rate by 25bps or more at the June 2027 FOMC meeting",
    ]),
    dict(label="2026 US real GDP year-over-year growth", date="2027-01-30", outcomes=[
        "US real GDP growth for calendar 2026 (BEA initial estimate) is below 1.5%",
        "US real GDP growth for calendar 2026 is between 1.5% and 2.5%",
        "US real GDP growth for calendar 2026 is between 2.5% and 3.5%",
        "US real GDP growth for calendar 2026 is above 3.5%",
    ]),
    dict(label="EUR/USD year-end 2026", date="2026-12-31", outcomes=[
        "the EUR/USD spot rate at year-end 2026 is below 1.05",
        "the EUR/USD spot rate at year-end 2026 is between 1.05 and 1.15",
        "the EUR/USD spot rate at year-end 2026 is above 1.15",
    ]),
    dict(label="WTI crude oil year-end 2026 price bucket", date="2026-12-31", outcomes=[
        "the front-month WTI crude oil futures settlement on the last trading day of 2026 is below $60/bbl",
        "the front-month WTI crude oil futures settlement on the last trading day of 2026 is between $60 and $80/bbl",
        "the front-month WTI crude oil futures settlement on the last trading day of 2026 is above $80/bbl",
    ]),
    dict(label="Gold price year-end 2026", date="2026-12-31", outcomes=[
        "the LBMA AM gold fix on the last trading day of 2026 is below $2,500/oz",
        "the LBMA AM gold fix on the last trading day of 2026 is between $2,500 and $3,000/oz",
        "the LBMA AM gold fix on the last trading day of 2026 is above $3,000/oz",
    ]),
    dict(label="US 10-year Treasury yield year-end 2026", date="2026-12-31", outcomes=[
        "the 10-year US Treasury constant-maturity yield on the last business day of 2026 is below 3.5%",
        "the 10-year US Treasury constant-maturity yield on the last business day of 2026 is between 3.5% and 4.5%",
        "the 10-year US Treasury constant-maturity yield on the last business day of 2026 is above 4.5%",
    ]),
    dict(label="UK CPI YoY November 2026", date="2026-12-17", outcomes=[
        "UK headline CPI year-over-year for November 2026 is below 2.0%",
        "UK headline CPI year-over-year for November 2026 is between 2.0% and 3.0%",
        "UK headline CPI year-over-year for November 2026 is above 3.0%",
    ]),
    dict(label="ECB main refinancing rate, December 2026", date="2026-12-31", outcomes=[
        "the ECB main refinancing operations rate on 31 December 2026 is below 2.00%",
        "the ECB main refinancing operations rate on 31 December 2026 is between 2.00% and 3.00%",
        "the ECB main refinancing operations rate on 31 December 2026 is above 3.00%",
    ]),
    # ----- Sports (10) -----
    dict(label="2026 World Series winner league", date="2026-11-15", outcomes=[
        "the 2026 World Series is won by an American League team",
        "the 2026 World Series is won by a National League team",
    ]),
    dict(label="2026 NHL Stanley Cup winner conference", date="2026-06-30", outcomes=[
        "the 2026 NHL Stanley Cup is won by an Eastern Conference team",
        "the 2026 NHL Stanley Cup is won by a Western Conference team",
    ]),
    dict(label="2027 Australian Open men's singles winner", date="2027-01-31", outcomes=[
        "Carlos Alcaraz wins the 2027 Australian Open men's singles championship",
        "Jannik Sinner wins the 2027 Australian Open men's singles championship",
        "Novak Djokovic wins the 2027 Australian Open men's singles championship",
        "the 2027 Australian Open men's singles championship is won by a different player",
    ]),
    dict(label="2027 Six Nations rugby winner", date="2027-03-31", outcomes=[
        "the 2027 Six Nations rugby championship is won by Ireland or Wales",
        "the 2027 Six Nations rugby championship is won by France or Italy",
        "the 2027 Six Nations rugby championship is won by England or Scotland",
    ]),
    dict(label="2026 F1 Constructors' Championship winner", date="2026-12-15", outcomes=[
        "the 2026 Formula 1 Constructors' Championship is won by Red Bull Racing",
        "the 2026 Formula 1 Constructors' Championship is won by Mercedes",
        "the 2026 Formula 1 Constructors' Championship is won by Ferrari",
        "the 2026 Formula 1 Constructors' Championship is won by McLaren or another constructor",
    ]),
    dict(label="2026 F1 Drivers' Championship winner", date="2026-12-15", outcomes=[
        "the 2026 Formula 1 Drivers' Championship is won by a Red Bull Racing driver",
        "the 2026 Formula 1 Drivers' Championship is won by a McLaren driver",
        "the 2026 Formula 1 Drivers' Championship is won by a Ferrari driver",
        "the 2026 Formula 1 Drivers' Championship is won by a driver from another constructor",
    ]),
    dict(label="2026 Tour de France winner nationality", date="2026-07-26", outcomes=[
        "the 2026 Tour de France is won by a rider from Slovenia or Denmark",
        "the 2026 Tour de France is won by a rider from France or Belgium",
        "the 2026 Tour de France is won by a rider from a different country",
    ]),
    dict(label="2027 ICC T20 Cricket World Cup winner region", date="2027-06-30", outcomes=[
        "the 2027 ICC T20 Men's Cricket World Cup is won by a team from South Asia (India, Pakistan, Sri Lanka, Bangladesh, Afghanistan)",
        "the 2027 ICC T20 Men's Cricket World Cup is won by a team from Australia, England, or New Zealand",
        "the 2027 ICC T20 Men's Cricket World Cup is won by a team from another region",
    ]),
    dict(label="2026 NCAA Division I football national champion conference", date="2027-01-15", outcomes=[
        "the 2026 NCAA Division I football national champion is from the SEC",
        "the 2026 NCAA Division I football national champion is from the Big Ten",
        "the 2026 NCAA Division I football national champion is from another conference",
    ]),
    dict(label="2026 PGA Tour FedEx Cup winner nationality", date="2026-09-15", outcomes=[
        "the 2026 PGA Tour FedEx Cup champion is American",
        "the 2026 PGA Tour FedEx Cup champion is from another country",
    ]),
    # ----- Awards / culture (10) -----
    dict(label="2026 Emmy Outstanding Drama Series platform", date="2026-09-20", outcomes=[
        "the 2026 Primetime Emmy for Outstanding Drama Series goes to a series airing on a streaming platform (Netflix, Apple TV+, Amazon, Disney+, etc.)",
        "the 2026 Primetime Emmy for Outstanding Drama Series goes to a series airing on a premium-cable network (HBO, AMC, Showtime, etc.)",
        "the 2026 Primetime Emmy for Outstanding Drama Series goes to a series airing on broadcast television",
    ]),
    dict(label="2026 Tony Best Musical genre", date="2026-06-15", outcomes=[
        "the 2026 Tony Award for Best Musical is won by an original musical (not based on a film, book, or other prior work)",
        "the 2026 Tony Award for Best Musical is won by a stage adaptation of a film",
        "the 2026 Tony Award for Best Musical is won by a stage adaptation of any other prior work",
    ]),
    dict(label="2027 BAFTA Best Film origin", date="2027-02-21", outcomes=[
        "the 2027 BAFTA Best Film award goes to a film with primarily British production",
        "the 2027 BAFTA Best Film award goes to a film with primarily American production",
        "the 2027 BAFTA Best Film award goes to a film with primarily other-country production",
    ]),
    dict(label="2026 Cannes Palme d'Or winner nationality", date="2026-05-25", outcomes=[
        "the 2026 Cannes Palme d'Or is awarded to a film by a European director",
        "the 2026 Cannes Palme d'Or is awarded to a film by an Asian director",
        "the 2026 Cannes Palme d'Or is awarded to a film by a director from another region",
    ]),
    dict(label="2027 Hugo Award Best Novel author origin", date="2027-08-31", outcomes=[
        "the 2027 Hugo Award for Best Novel is won by an author from the United States",
        "the 2027 Hugo Award for Best Novel is won by an author from the United Kingdom",
        "the 2027 Hugo Award for Best Novel is won by an author from a different country",
    ]),
    dict(label="2026 Mercury Music Prize winner gender", date="2026-09-30", outcomes=[
        "the 2026 Mercury Music Prize is won by a male solo artist or all-male group",
        "the 2026 Mercury Music Prize is won by a female solo artist or all-female group",
        "the 2026 Mercury Music Prize is won by a mixed-gender group or non-binary artist",
    ]),
    dict(label="2027 Berlin Golden Bear winner type", date="2027-02-28", outcomes=[
        "the 2027 Berlin Film Festival Golden Bear is awarded to a fiction feature",
        "the 2027 Berlin Film Festival Golden Bear is awarded to a documentary",
        "the 2027 Berlin Film Festival Golden Bear is not awarded or is awarded ex aequo",
    ]),
    dict(label="2026 Olivier Best New Play origin", date="2026-04-15", outcomes=[
        "the 2026 Olivier Award for Best New Play is won by a play written by a UK-based playwright",
        "the 2026 Olivier Award for Best New Play is won by a play written by a non-UK playwright",
        "the 2026 Olivier Award for Best New Play category is not awarded in 2026",
    ]),
    dict(label="2026 Nobel Literature laureate region", date="2026-12-10", outcomes=[
        "the 2026 Nobel Prize in Literature is awarded to an author from Europe",
        "the 2026 Nobel Prize in Literature is awarded to an author from the Americas",
        "the 2026 Nobel Prize in Literature is awarded to an author from Asia, Africa, or Oceania",
    ]),
    dict(label="2026 International Booker Prize winner language", date="2026-05-31", outcomes=[
        "the 2026 International Booker Prize is awarded to a translation from a European language",
        "the 2026 International Booker Prize is awarded to a translation from an Asian language",
        "the 2026 International Booker Prize is awarded to a translation from another world region",
    ]),
    # ----- Tech (10) -----
    dict(label="Top GPQA score, year-end 2026", date="2026-12-31", outcomes=[
        "the highest publicly reported GPQA Diamond score on 31 December 2026 is held by an Anthropic Claude model",
        "the highest publicly reported GPQA Diamond score on 31 December 2026 is held by an OpenAI GPT model",
        "the highest publicly reported GPQA Diamond score on 31 December 2026 is held by a Google Gemini model",
        "the highest publicly reported GPQA Diamond score on 31 December 2026 is held by some other model",
    ]),
    dict(label="Top HumanEval+ score, year-end 2026", date="2026-12-31", outcomes=[
        "the highest publicly reported HumanEval+ score on 31 December 2026 is held by an Anthropic Claude model",
        "the highest publicly reported HumanEval+ score on 31 December 2026 is held by an OpenAI GPT model",
        "the highest publicly reported HumanEval+ score on 31 December 2026 is held by a Google Gemini model",
        "the highest publicly reported HumanEval+ score on 31 December 2026 is held by some other model",
    ]),
    dict(label="Largest US AI startup IPO 2026", date="2026-12-31", outcomes=[
        "the largest US AI-startup IPO of 2026 (by initial market capitalisation) is from an AI infrastructure / chip company",
        "the largest US AI-startup IPO of 2026 is from an AI model / foundation-model company",
        "the largest US AI-startup IPO of 2026 is from an AI applications company",
        "the largest US AI-startup IPO of 2026 is from a different category, or no AI-startup IPO of significant size occurs in 2026",
    ]),
    dict(label="Top global cloud provider revenue share 2026", date="2027-02-15", outcomes=[
        "Amazon Web Services has the largest 2026 annual revenue share in the global public-cloud market (IaaS+PaaS)",
        "Microsoft Azure has the largest 2026 annual revenue share in the global public-cloud market",
        "Google Cloud has the largest 2026 annual revenue share in the global public-cloud market",
        "another provider has the largest 2026 annual revenue share",
    ]),
    dict(label="Year-end 2026 leading mobile OS", date="2026-12-31", outcomes=[
        "Android has a larger global smartphone OS market share at year-end 2026",
        "iOS has a larger global smartphone OS market share at year-end 2026",
        "any other mobile OS has the largest market share at year-end 2026",
    ]),
    dict(label="Top open-source LLM family year-end 2026", date="2026-12-31", outcomes=[
        "Meta's Llama family is the most-downloaded open-source LLM family on Hugging Face for calendar 2026",
        "Mistral's family is the most-downloaded open-source LLM family on Hugging Face for calendar 2026",
        "Alibaba's Qwen family is the most-downloaded open-source LLM family on Hugging Face for calendar 2026",
        "another open-source LLM family is the most-downloaded for calendar 2026",
    ]),
    dict(label="Apple WWDC 2026 keynote", date="2026-06-15", outcomes=[
        "the WWDC 2026 keynote (June 2026) features a major new Apple AI / Apple Intelligence announcement",
        "the WWDC 2026 keynote features a major new Apple operating-system or developer-tools announcement",
        "the WWDC 2026 keynote features a major new Apple hardware announcement",
    ]),
    dict(label="Tesla 2026 Q4 deliveries bucket", date="2027-01-31", outcomes=[
        "Tesla's Q4 2026 vehicle deliveries are below 450,000",
        "Tesla's Q4 2026 vehicle deliveries are between 450,000 and 550,000",
        "Tesla's Q4 2026 vehicle deliveries are above 550,000",
    ]),
    dict(label="Top semiconductor revenue 2026: foundry vs IDM", date="2027-02-15", outcomes=[
        "the world's largest semiconductor company by 2026 revenue is a pure-play foundry (e.g., TSMC)",
        "the world's largest semiconductor company by 2026 revenue is a fabless designer (e.g., Nvidia, Qualcomm)",
        "the world's largest semiconductor company by 2026 revenue is an integrated device manufacturer (e.g., Samsung, Intel)",
    ]),
    dict(label="Largest 2026 US data-center investment site", date="2026-12-31", outcomes=[
        "the largest US AI / hyperscaler data-center investment announced or opened in 2026 (by capex) is in Texas",
        "the largest US AI / hyperscaler data-center investment announced or opened in 2026 (by capex) is in Virginia",
        "the largest US AI / hyperscaler data-center investment announced or opened in 2026 is in another US state",
    ]),
    # ----- Nobel / Science (8) -----
    dict(label="2026 Nobel Medicine laureate field", date="2026-10-05", outcomes=[
        "the 2026 Nobel Prize in Physiology or Medicine recognises work primarily in immunology",
        "the 2026 Nobel Prize in Physiology or Medicine recognises work primarily in genetics or molecular biology",
        "the 2026 Nobel Prize in Physiology or Medicine recognises work primarily in neuroscience",
        "the 2026 Nobel Prize in Physiology or Medicine recognises work in a different biomedical area",
    ]),
    dict(label="2026 Nobel Economics laureate field", date="2026-10-12", outcomes=[
        "the 2026 Sveriges Riksbank Prize in Economic Sciences (Nobel Economics) recognises work primarily in macroeconomics or monetary economics",
        "the 2026 Nobel Economics Prize recognises work primarily in microeconomics, game theory, or mechanism design",
        "the 2026 Nobel Economics Prize recognises work primarily in development, labour, or applied economics",
        "the 2026 Nobel Economics Prize recognises work in another subfield",
    ]),
    dict(label="2026 Wolf Prize in Mathematics", date="2026-06-01", outcomes=[
        "the 2026 Wolf Prize in Mathematics recognises work primarily in pure mathematics (analysis, algebra, geometry, topology, number theory)",
        "the 2026 Wolf Prize in Mathematics recognises work primarily in applied mathematics or mathematical physics",
        "the 2026 Wolf Prize in Mathematics is not awarded in 2026",
    ]),
    dict(label="2027 Breakthrough Prize Life Sciences focus", date="2027-04-30", outcomes=[
        "the 2027 Breakthrough Prize in Life Sciences (announced in late 2026 / early 2027) recognises work primarily in cancer biology",
        "the 2027 Breakthrough Prize in Life Sciences recognises work primarily in neuroscience",
        "the 2027 Breakthrough Prize in Life Sciences recognises work primarily in genetics or genome editing",
        "the 2027 Breakthrough Prize in Life Sciences recognises work in another biomedical area",
    ]),
    dict(label="2026 IMO top country", date="2026-07-31", outcomes=[
        "the team-score winner of the 2026 International Mathematical Olympiad is China",
        "the team-score winner of the 2026 International Mathematical Olympiad is the United States",
        "the team-score winner of the 2026 International Mathematical Olympiad is another country",
    ]),
    dict(label="2026 IPhO top country", date="2026-08-15", outcomes=[
        "the highest-scoring team at the 2026 International Physics Olympiad is China",
        "the highest-scoring team at the 2026 International Physics Olympiad is the United States or Russia",
        "the highest-scoring team at the 2026 International Physics Olympiad is another country",
    ]),
    dict(label="ITER first plasma timing", date="2027-12-31", outcomes=[
        "ITER achieves first plasma in 2026",
        "ITER achieves first plasma in 2027",
        "ITER does not achieve first plasma by 31 December 2027",
    ]),
    dict(label="2026 ACM Gödel Prize area", date="2026-07-15", outcomes=[
        "the 2026 ACM Gödel Prize recognises work in algorithms or complexity theory",
        "the 2026 ACM Gödel Prize recognises work in cryptography or computational learning",
        "the 2026 ACM Gödel Prize recognises work in another area of theoretical computer science",
    ]),
    # ----- Health / medicine (5) -----
    dict(label="FDA Alzheimer's drug action 2026", date="2026-12-31", outcomes=[
        "the FDA grants traditional or accelerated approval to at least one new Alzheimer's-disease therapy in 2026",
        "the FDA issues a major label expansion or restriction for an existing approved Alzheimer's therapy in 2026",
        "the FDA takes no significant approval or label action on Alzheimer's-disease therapies in 2026",
    ]),
    dict(label="GLP-1 agonist label expansion 2026", date="2026-12-31", outcomes=[
        "an FDA label expansion approves a GLP-1 receptor agonist for a new chronic-disease indication (e.g., kidney disease, NASH, sleep apnoea, addiction) in 2026",
        "no major new chronic-disease indication is approved for any GLP-1 receptor agonist in 2026",
        "the FDA issues a major safety restriction on GLP-1 receptor agonists in 2026",
    ]),
    dict(label="WHO PHEIC declarations 2026", date="2026-12-31", outcomes=[
        "the WHO declares a new Public Health Emergency of International Concern (PHEIC) in calendar 2026",
        "the WHO maintains an existing PHEIC throughout 2026 without declaring a new one",
        "the WHO has no active PHEIC for any portion of calendar 2026",
    ]),
    dict(label="Cancer screening trial readout 2026", date="2026-12-31", outcomes=[
        "a major (>5,000 participant) randomised cancer-screening trial reports a significant mortality benefit in 2026",
        "a major randomised cancer-screening trial reports a null or modest result in 2026",
        "no major randomised cancer-screening trial reports primary results in 2026",
    ]),
    dict(label="2026 mpox global status", date="2026-12-31", outcomes=[
        "global monthly confirmed mpox cases in December 2026 are below 1,000",
        "global monthly confirmed mpox cases in December 2026 are between 1,000 and 5,000",
        "global monthly confirmed mpox cases in December 2026 are above 5,000",
    ]),
    # ----- Climate / environment (3) -----
    dict(label="2026 global temperature ranking", date="2027-01-31", outcomes=[
        "calendar 2026 is the warmest year on record (NASA GISTEMP / NOAA / Copernicus, ranked #1)",
        "calendar 2026 is the second-warmest year on record",
        "calendar 2026 is third-warmest or cooler",
    ]),
    dict(label="2026 Atlantic hurricane season activity", date="2026-12-01", outcomes=[
        "the 2026 Atlantic hurricane season has fewer than 12 named storms",
        "the 2026 Atlantic hurricane season has between 12 and 17 named storms",
        "the 2026 Atlantic hurricane season has 18 or more named storms",
    ]),
    dict(label="2026 Arctic sea ice minimum extent", date="2026-09-30", outcomes=[
        "the 2026 Arctic sea ice minimum extent is below 4.0 million sq km",
        "the 2026 Arctic sea ice minimum extent is between 4.0 and 4.7 million sq km",
        "the 2026 Arctic sea ice minimum extent is above 4.7 million sq km",
    ]),
]


def make_question(qid: str, outcome_text: str, date: str) -> PalekaQuestion:
    return PalekaQuestion(
        id=qid,
        title=outcome_text,
        body=outcome_text,
        resolution_date=date,
        question_type="binary",
        data_source="real_agent_case_study",
        url=None,
        resolution=None,
    )


def build_specialists() -> list:
    """Return 4 specialist LLM clients."""
    return [
        AnthropicClient(model="claude-haiku-4-5-20251001"),
        AzureGPT54Client(deployment_env="AZURE_OPENAI_DEPLOYMENT_MINI"),
        AzureGPT54Client(deployment_env="AZURE_OPENAI_DEPLOYMENT_NANO"),
        GroqClient(model="llama-3.3-70b-versatile", api_key_env="GROQ_API_KEY"),
    ]


def project_partition(p: np.ndarray) -> tuple[np.ndarray, float]:
    """Project p onto the simplex (sum=1, p>=0); return (proj, eps^*)."""
    m = len(p)
    clique = Clique(m=m, relations=[Relation(type="partition", indices=tuple(range(m)))])
    proj = jcd_project(clique, p)
    eps = float(np.linalg.norm(p - proj))
    return proj, eps


def main(args) -> None:
    rng = random.Random(SEED)
    specialists = build_specialists()
    short_names = ["Claude-Haiku", "GPT-5.4-mini", "GPT-5.4-nano", "Llama-3.3-70b"]

    results = []
    for partition in PARTITIONS:
        m = len(partition["outcomes"])
        # Random specialist assignment, one per outcome (sample without
        # replacement when feasible to maximize cross-LLM mixing).
        if m <= len(specialists):
            assign = rng.sample(range(len(specialists)), m)
        else:
            assign = [rng.randrange(len(specialists)) for _ in range(m)]

        per_outcome_means = []
        per_outcome_samples = []
        per_outcome_specialist = []
        for j, outcome_text in enumerate(partition["outcomes"]):
            sp_idx = assign[j]
            sp = specialists[sp_idx]
            q = make_question(
                qid=f"{partition['label']}::outcome{j}",
                outcome_text=f"What is the probability that {outcome_text}?",
                date=partition["date"],
            )
            samples = sp.forecast(q, K, temperature=TEMPERATURE)
            if len(samples) == 0:
                print(f"  WARN: all K samples failed for {short_names[sp_idx]} on outcome {j}; "
                      "imputing 0.5")
                mean = 0.5
            else:
                mean = float(np.mean(samples))
            per_outcome_means.append(mean)
            per_outcome_samples.append(samples.tolist())
            per_outcome_specialist.append(short_names[sp_idx])
            print(f"  {short_names[sp_idx]:14s} | outcome {j} | "
                  f"K={len(samples)}/{K} | mean={mean:.3f}")

        p = np.array(per_outcome_means, dtype=float)
        proj, eps = project_partition(p)
        sum_violation = abs(p.sum() - 1.0)

        print(f"\n  partition: {partition['label']}")
        print(f"    raw quote      = {[round(x,3) for x in p.tolist()]}")
        print(f"    sum            = {p.sum():.3f}  (|sum-1| = {sum_violation:.3f})")
        print(f"    projected      = {[round(x,3) for x in proj.tolist()]}")
        print(f"    eps^*          = {eps:.4f}\n")

        results.append(dict(
            label=partition["label"],
            outcomes=partition["outcomes"],
            assigned_specialists=per_outcome_specialist,
            per_outcome_means=per_outcome_means,
            per_outcome_samples=per_outcome_samples,
            sum=p.sum(),
            sum_violation=sum_violation,
            eps_star=eps,
            projected=proj.tolist(),
        ))

    # ----- Aggregate report -----
    eps_arr = np.array([r["eps_star"] for r in results])
    sumv_arr = np.array([r["sum_violation"] for r in results])
    print("=" * 72)
    print("REAL-AGENT CASE STUDY: SUMMARY")
    print("=" * 72)
    print(f"Partitions tested      : {len(results)}")
    print(f"Partitions with eps*>0 : {int(np.sum(eps_arr > 1e-6))}/{len(results)}")
    print(f"Mean eps^*             : {eps_arr.mean():.4f}")
    print(f"Max  eps^*             : {eps_arr.max():.4f}  ({results[int(np.argmax(eps_arr))]['label']})")
    print(f"Mean |sum - 1|         : {sumv_arr.mean():.4f}")
    print(f"Max  |sum - 1|         : {sumv_arr.max():.4f}")

    # Per-partition table
    print(f"\n{'partition':<48s}{'sum':>8s}{'eps*':>10s}")
    print("-" * 66)
    for r in results:
        print(f"{r['label'][:46]:<48s}{r['sum']:>8.3f}{r['eps_star']:>10.4f}")

    # Save raw results
    out = Path(__file__).resolve().parent.parent / "real_agent_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=K, help="samples per (LLM, outcome)")
    args = parser.parse_args()
    main(args)
