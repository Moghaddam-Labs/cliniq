"""
ClinIQ Dashboard — Data layer
get_portfolio_risk: ranked site risk summaries with alert flags and caching.
get_site_kpi_timeseries, get_trials, get_trial_sites, get_site_deviations:
cached lookups used by the dashboard and the AI assistant.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from cliniq.analytics.velocity import compute_velocity
from cliniq.dashboard.cache import cache_get, cache_set
from cliniq.db.models import (
    ProtocolDeviation, RiskScore, SiteKPISnapshot, Trial, TrialSite,
)
from cliniq.ml.risk_model import compute_risk_score


@dataclass
class SiteRiskSummary:
    trial_site_id: int
    site_id: str
    site_name: str
    country: str
    composite_score: float
    dropout_probability: float
    enrolment_pct: Optional[float]
    velocity_28d: Optional[float]
    is_on_track: Optional[bool]
    alert_flags: list[str]


def get_portfolio_risk(db: Session, trial_id: int,
                       as_of: Optional[date] = None) -> list[SiteRiskSummary]:
    """Ranked list of site risk summaries for a trial, with caching."""
    today = as_of or date.today()
    cache_key = f"portfolio_risk:{trial_id}:{today}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    trial_sites = (
        db.query(TrialSite)
        .filter(TrialSite.trial_id == trial_id, TrialSite.is_active.is_(True))
        .all()
    )

    summaries: list[SiteRiskSummary] = []
    for ts in trial_sites:
        risk_row = (
            db.query(RiskScore)
            .filter_by(trial_site_id=ts.id, score_date=today)
            .first()
        )
        if risk_row:
            composite = risk_row.composite_score
            dropout   = risk_row.dropout_probability or 0.0
        else:
            r = compute_risk_score(db, ts.id, today)
            composite = r.composite_score
            dropout   = r.dropout_probability

        snap = (
            db.query(SiteKPISnapshot)
            .filter_by(trial_site_id=ts.id)
            .order_by(SiteKPISnapshot.snapshot_date.desc())
            .first()
        )
        enrolment_pct = snap.enrolment_pct if snap else None
        velocity_result = compute_velocity(db, ts.id, today)
        velocity_28d    = velocity_result.velocity_28d
        is_on_track     = velocity_result.is_on_track

        flags = []
        if composite >= 70:   flags.append("HIGH_RISK")
        if dropout >= 0.65:   flags.append("DROPOUT_RISK")
        if snap and snap.data_lag_mean and snap.data_lag_mean >= 14:
            flags.append("DATA_LAG")
        if not is_on_track:   flags.append("OFF_TRACK")

        summaries.append(SiteRiskSummary(
            trial_site_id=ts.id,
            site_id=ts.site.site_id,
            site_name=ts.site.name,
            country=ts.site.country,
            composite_score=composite,
            dropout_probability=dropout,
            enrolment_pct=enrolment_pct,
            velocity_28d=velocity_28d,
            is_on_track=is_on_track,
            alert_flags=flags,
        ))

    summaries.sort(key=lambda s: s.composite_score, reverse=True)
    cache_set(cache_key, summaries)
    return summaries


def get_site_kpi_timeseries(db: Session, trial_site_id: int,
                            days: int = 90) -> list[SiteKPISnapshot]:
    """KPI snapshots for one trial site over the last `days` days, oldest first."""
    cache_key = f"kpi_timeseries:{trial_site_id}:{days}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    since = date.today() - timedelta(days=days)
    snaps = (
        db.query(SiteKPISnapshot)
        .filter(
            SiteKPISnapshot.trial_site_id == trial_site_id,
            SiteKPISnapshot.snapshot_date >= since,
        )
        .order_by(SiteKPISnapshot.snapshot_date.asc())
        .all()
    )
    cache_set(cache_key, snaps)
    return snaps


def get_trials(db: Session) -> list[Trial]:
    """All trials, ordered by title, for dropdowns and portfolio views."""
    cache_key = "trials:all"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    trials = db.query(Trial).order_by(Trial.title.asc()).all()
    cache_set(cache_key, trials)
    return trials


def get_trial_sites(db: Session, trial_id: int) -> list[TrialSite]:
    """Active sites for a trial."""
    cache_key = f"trial_sites:{trial_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    trial_sites = (
        db.query(TrialSite)
        .filter(TrialSite.trial_id == trial_id, TrialSite.is_active.is_(True))
        .order_by(TrialSite.id.asc())
        .all()
    )
    cache_set(cache_key, trial_sites)
    return trial_sites


def get_site_deviations(db: Session, trial_site_id: int,
                        limit: int = 50) -> list[ProtocolDeviation]:
    """Most recent protocol deviations for a trial site, newest first.

    Not cached: deviations are reviewed and resolved live, so the list
    should always reflect the current state.
    """
    return (
        db.query(ProtocolDeviation)
        .filter(ProtocolDeviation.trial_site_id == trial_site_id)
        .order_by(ProtocolDeviation.deviation_date.desc())
        .limit(limit)
        .all()
    )
