#!/usr/bin/env python3
"""
TNC Hinglish ASR - Levenshtein Word Error Rate (WER) & CER Benchmark Tool
Evaluates model accuracy on Hindi, Indian-English, and Hinglish transcripts.
"""

import sys
import argparse

def compute_levenshtein(r, h):
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1] + 1, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    i, j = m, n
    subs, dels, inss = 0, 0, 0
    ops = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and r[i - 1] == h[j - 1]:
            ops.insert(0, ("match", r[i - 1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.insert(0, ("sub", r[i - 1], h[j - 1]))
            subs += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.insert(0, ("del", r[i - 1]))
            dels += 1; i -= 1
        else:
            ops.insert(0, ("ins", h[j - 1]))
            inss += 1; j -= 1

    return dp[m][n], subs, dels, inss, ops

def evaluate_wer_cer(reference, hypothesis):
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    dist_w, subs, dels, inss, ops = compute_levenshtein(ref_words, hyp_words)
    wer_pct = (dist_w / max(1, len(ref_words))) * 100.0

    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    dist_c, _, _, _, _ = compute_levenshtein(ref_chars, hyp_chars)
    cer_pct = (dist_c / max(1, len(ref_chars))) * 100.0

    print("=== TNC Hinglish ASR Evaluation ===")
    print(f"Reference  (Ground Truth): '{reference}'")
    print(f"Hypothesis (Model Output): '{hypothesis}'\n")
    print(f"  • Word Error Rate (WER) : {wer_pct:5.1f}% (Sub: {subs}, Del: {dels}, Ins: {inss})")
    print(f"  • Char Error Rate (CER) : {cer_pct:5.1f}% (Char Edits: {dist_c})")
    print(f"  • Accuracy Match Score  : {max(0, 100 - wer_pct):5.1f}%\n")

    diff_str = ""
    for op in ops:
        if op[0] == "match":
            diff_str += f"\033[32m{op[1]} \033[0m"
        elif op[0] == "sub":
            diff_str += f"\033[33m[{op[1]}➔{op[2]}] \033[0m"
        elif op[0] == "del":
            diff_str += f"\033[31m-{op[1]} \033[0m"
        elif op[0] == "ins":
            diff_str += f"\033[36m+{op[1]} \033[0m"

    print(f"Levenshtein Diff: {diff_str}\n")
    return wer_pct, cer_pct

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TNC ASR Evaluation Benchmark")
    parser.add_argument("--ref", type=str, default="today ki meeting 3 baje start hogi", help="Reference ground truth text")
    parser.add_argument("--hyp", type=str, default="today ki meeting 3 baje start hogi", help="Model hypothesis transcript")

    args = parser.parse_args()
    evaluate_wer_cer(args.ref, args.hyp)
