def ensemble_threat_score(anomaly_score: float, ip_risk_score: float) -> float:
    """
    Combine anomaly score and IP risk into a single threat score.
    Escalates on toxic IPs, never depresses the anomaly score.
    """
    base = float(anomaly_score)
    if ip_risk_score >= 1.0:
        # Escalate 25% toward 1.0
        threat_score = min(1.0, base + 0.25 * (1.0 - base))
    else:
        threat_score = base

    # Clamp within [0, 1]
    return max(0.0, min(threat_score, 1.0))
