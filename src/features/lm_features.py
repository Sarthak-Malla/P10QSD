"""
lm_features.py

Loughran-McDonald (2011) finance-specific sentiment features.

VADER was built for social media and misclassifies ~40% of finance terms.
LM was constructed from 10-K filings and is the gold standard in finance NLP.

Key categories:
  LM_NEGATIVE     - words signaling bad outcomes (loss, impair, adverse, default...)
  LM_UNCERTAINTY  - hedging language (may, could, uncertain, approximately...)
  LM_LITIGIOUS    - legal/regulatory risk (lawsuit, penalty, violation, breach...)
  LM_POSITIVE     - words signaling good outcomes (grow, achieve, record, gain...)
  LM_CONSTRAINING - words indicating limitations (require, must, shall, limit...)

Proven in literature: LM negative word count is the strongest text predictor
of negative stock returns (Loughran & McDonald, 2011; Li, 2010).

The DELTA features (change from previous quarter) are even more powerful:
a company that adds negative/uncertainty words is sending a signal.
"""

import re
import numpy as np
import pandas as pd

# ── Loughran-McDonald word sets ────────────────────────────────────────────────
# Source: Loughran & McDonald (2011) master dictionary, most frequent terms

LM_NEGATIVE = {
    "abandon","abnormal","abolish","abrupt","absence","abuse","adverse","adversely",
    "allegations","alleged","annulment","appeal","arbitrary","bankrupt","bankruptcy",
    "breach","burden","cancel","cancellation","casualty","challenge","charges",
    "claims","closure","collapse","complaint","concern","concerned","conflict",
    "controversy","conviction","crisis","curtail","damage","damages","decline",
    "decrease","decreasing","default","deficit","delay","delinquent","deny",
    "deteriorate","deteriorating","difficult","difficulty","disappoint","disaster",
    "dispute","disruption","distress","doubt","downgrade","eliminate","enforcement",
    "erode","eviction","exceed","exhaust","fail","failed","failure","falling",
    "fine","fines","force","forfeit","fraud","harm","harmed","harmful","illegal",
    "impair","impaired","impairment","impede","inability","inadequate","incident",
    "injunction","insufficient","investigation","involuntary","judgment","lawsuit",
    "lawsuits","layoff","layoffs","liability","liabilities","limitation","liquidation",
    "litigant","litigation","loss","losses","material","mislead","negative","neglect",
    "negligence","noncompliance","objection","obstacle","oppose","opposition",
    "outage","penalty","penalties","problem","problems","prohibit","recession",
    "reduced","reduction","refuse","reject","rejected","restructure","restructuring",
    "restate","restatement","risk","risks","sanction","sanctions","shortfall",
    "suspension","terminate","termination","threat","threatened","uncertain",
    "unfavorable","unforeseen","violation","violations","warning","weakness",
    "weaknesses","writedown","writeoff","wrong","worsen","worsening",
    "decline","declining","depreciation","deterioration","downward","dropping",
    "eroding","exceeding","failing","impacting","inadequacy","insolvent",
    "insufficient","interrupt","interruption","jeopardize","limited","lost",
    "negative","notwithstanding","overcome","overstatement","restrict","restricted",
    "restriction","severe","significantly","volatile","volatility",
}

LM_UNCERTAINTY = {
    "approximately","around","believe","contingent","could","depend","dependent",
    "difficult","doubt","estimate","evaluate","eventual","examine","expect",
    "fluctuate","fluctuation","generally","impact","imprecise","indefinite",
    "indefinitely","likely","may","might","obscure","pending","possibly","predict",
    "probable","roughly","seem","should","sometimes","suggest","uncertain",
    "uncertainty","unclear","unknown","unpredictable","unusual","unstable",
    "variable","vague","various","varies","vary","whether","although","assume",
    "assumed","assumption","assumptions","approximately","believe","considered",
    "depends","estimated","expected","likely","may","might","projected","should",
    "subject","subjective","suppose","uncertain","unforeseen","unpredictable",
}

LM_LITIGIOUS = {
    "allegation","alleged","appeal","arbitration","attorney","attorneys","breach",
    "claim","claims","class","complaint","compliance","consent","conviction",
    "counsel","court","courts","defendant","defendants","dispute","disputes",
    "enforcement","evidence","filing","fraud","government","indictment","injunction",
    "investigate","investigation","judgment","jury","lawsuit","lawsuits","legal",
    "liable","liabilities","litigation","litigations","negligence","penalty",
    "penalties","plaintiff","plaintiffs","proceeding","proceedings","prosecution",
    "regulatory","regulator","regulators","sanction","sanctions","settlement",
    "settlements","statute","subpoena","sue","suit","suits","tribunal","verdict",
    "violation","violations","warrant","class-action","enforcement","felony",
    "fine","fines","illegal","infringement","misconduct","noncompliance","offense",
}

LM_POSITIVE = {
    "achieve","achieved","acquisition","advantage","benefit","capability","confident",
    "confidence","deliver","delivered","effective","efficient","enhance","enhanced",
    "excellent","expand","expanded","gain","gains","good","grow","growing","growth",
    "high","improve","improved","improving","increase","increased","innovative",
    "innovation","leadership","leverage","leveraged","opportunity","opportunities",
    "outperform","outperformed","positive","progress","progressed","promising",
    "record","revenue","revenues","reward","strength","strong","success","successful",
    "superior","support","value","advantage","awarded","best","breakthrough","broad",
    "capture","captured","competitive","complementary","comprehensive","demonstrated",
    "differentiated","drive","driven","enables","established","exceptional","exceed",
    "exceeded","expanding","experienced","favorable","growing","highest","leading",
    "momentum","obtain","organic","outstanding","peak","position","positioned",
    "premier","profitable","profitability","proven","realized","recognition",
    "return","robust","significant","stable","strong","substantial","successful",
}

LM_CONSTRAINING = {
    "require","required","requires","must","shall","mandatory","compelled","compulsory",
    "constraint","constrained","constrain","limit","limits","limited","limitation",
    "obligations","obligation","prohibit","prohibited","prohibition","regulate",
    "regulated","regulation","regulations","restricted","restriction","restrictions",
    "binding","covenant","covenants","compliance","comply","restriction","threshold",
}


def _tokenize(text: str) -> list:
    if not isinstance(text, str) or not text.strip():
        return []
    return re.findall(r"\b[a-z]+\b", text.lower())


def compute_lm_scores(text: str) -> dict:
    """
    Compute Loughran-McDonald finance sentiment scores.
    Returns ratios (word count / total words) for each category.
    Also returns net sentiment: (positive - negative) / (positive + negative + 1)
    """
    tokens = _tokenize(text)
    n = len(tokens)
    if n == 0:
        return {k: np.nan for k in [
            "lm_negative","lm_positive","lm_uncertainty",
            "lm_litigious","lm_constraining","lm_net_sentiment","lm_word_count"
        ]}

    neg  = sum(1 for t in tokens if t in LM_NEGATIVE)
    pos  = sum(1 for t in tokens if t in LM_POSITIVE)
    unc  = sum(1 for t in tokens if t in LM_UNCERTAINTY)
    lit  = sum(1 for t in tokens if t in LM_LITIGIOUS)
    con  = sum(1 for t in tokens if t in LM_CONSTRAINING)

    return {
        "lm_negative":     neg / n,
        "lm_positive":     pos / n,
        "lm_uncertainty":  unc / n,
        "lm_litigious":    lit / n,
        "lm_constraining": con / n,
        "lm_net_sentiment": (pos - neg) / (pos + neg + 1),
        "lm_word_count":   n,
    }


def compute_lm_delta(scores_series: pd.Series) -> pd.Series:
    """
    Compute quarter-over-quarter change for a given LM score series.
    Positive delta = score increased vs last quarter.
    For lm_negative: positive delta = more negative words added = bad signal.
    """
    return scores_series.diff()


def add_lm_features(filings_df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Add all LM features to a filings dataframe.
    Includes both level features and delta (change vs prev quarter) features.
    """
    df = filings_df.copy()

    # Level features
    lm_rows = df[text_col].apply(compute_lm_scores)
    lm_df = pd.DataFrame(lm_rows.tolist())
    df = pd.concat([df.reset_index(drop=True), lm_df.reset_index(drop=True)], axis=1)

    # Delta features (change vs previous quarter) - per company, already sorted by date
    for col in ["lm_negative","lm_positive","lm_uncertainty","lm_litigious"]:
        if col in df.columns:
            df[f"{col}_delta"] = compute_lm_delta(df[col])

    return df
