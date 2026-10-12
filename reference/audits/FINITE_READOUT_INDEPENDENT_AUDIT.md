# Original source record current readability independent audit

E2 SCOPED PASS for result revision r1. Reviewer: /root/source_readout_independent_audit, a separately authorized read-only reviewer and context. This report records the reviewer messages received on 2026-10-08; the producer transcribed the report and did not claim authorship of the independent calculation.

## Revision identity

- Producer code SHA256: d84d9ab4943540a1f52d15176afeb18a69e925c78cf1b998e319ea5d2055087e
- Plan SHA256: 455bddeb9c343ff7e47bb85ee6d9e7ba44c4cef4c88ec74b096fdca2b32155e3
- Source manifest SHA256: 53245273f9d3003fbbd2c1ce047d24fe00bcabb2160d75bb47cdbfe12f17f84a
- E0 SHA256: cb534b91ae5089b7e9614acc5586d6a28cc6bdbe19cc5c483ca26a6bcd336fb1
- Baseline gate SHA256: b3a8dfe02887c8c5e4b33dccb5d433f6f2bf2ab12603fcf3b8bba9ba34a96834
- Result revision manifest SHA256: 4513976d8f8ae21018d9ec8cc104b5673980c6596719004a1ed7b84c5c62cdca
- Independent corrected command digest, with terminal newline: 4fdf97d7d591ea193cb2bca40444d86e0ee010c42b6e519fb606c19a949c5c45

## Exact independent method

The reviewer imported no producer calculation functions. For spectral projectors P_x of rank d_x, set sigma_x=P_x/d_x and rho=sum_x pi_x sigma_x. Verify Tr(rho)=1 and that radical coefficients vanish.

For each screen S, independently compute M=S rho S by exact sparse Fraction multiplication, a_x=Tr(P_x S)/d_x and B_x=Tr(P_x M)=Tr(rho S P_x S). Use the historical incoming transition orientation T[x,y]. Define mu_y=sum_x pi_x T[x,y] a_x and nu_y=sum_x T[x,y] B_x.

For each pair of block indices a,b in [45,105,29], compute the numerator

sum_y {pi_y (S P_y S)[a,b] - mu_y ((S P_y)[a,b]+(P_y S)[a,b]) + nu_y P_y[a,b]}/d_y.

Independently compute g=2Tr(M)-2sum_y mu_y a_y, require equality with the archived original Gii, and divide each entry by g. This is a Heisenberg-gradient contraction on matrix units, not a call to producer focal_one. Full normalization is checked before compression; the 3 by 3 block is not renormalized.

## Exact result reproduction

All eight masks [0,5,8,13,16,21,24,29] have affine rank 3, hidden dimension 0, and same-mask fingerprint_only_readable true.

Exact matches comprise 288 compressed beta entries, 224 Y entries, 168 J entries, 128 A_m entries, 32 Gram norm diagonals, all block traces and observer bases. Every focal artifact hash matches RESULT_REVISION.json.

All eight full rho states are rational and have trace 1. All 32 compressed beta matrices are symmetric and positive semidefinite, checked through exact principal minors; their traces lie in [0,1]. All radical coefficients vanish. No compression renormalization occurs.

Supplemental row-normalized condition numbers, by mask, are 0:44.58285558181414, 5:21.01222329074534, 8:24.30077818852918, 13:22.110534568538355, 16:23.24518009948578, 21:21.480680575442836, 24:44.195762887070494, 29:24.020994935156615. These describe an ideal observer basis, not physiological signal-to-noise or stability.

## Endpoint de-alias witness

The separately computed rank 2 witness is retained in INDEPENDENT_E0_DEALIAS_WITNESS.json. Its nonconstant squared fingerprint equals p0 p1 and remains readable despite a hidden mixture direction. This establishes that the weaker fingerprint predicate is not an alias of rank 3. The reviewer used an independent Fraction implementation and imported no producer functions.

## Audit revision history

The first independent harness incorrectly substituted B_x=pi_x Tr(P_x S squared)/d_x, which requires an unestablished commutation relation between S and rho. Eight independent runs were produced, but the first beta comparison failed. The reviewer disclosed this audit-harness defect and corrected B_x to Tr(rho S P_x S). The corrected independent calculation exactly reproduced all eight r1 files. The producer script, plan, source manifest, witnesses, baseline and focal results were not patched. The initial independent comparison failure is not a scientific negative or a producer failure.

File modification times support the recorded chronology but are not tamper-proof: plan 16:23:17 JST; script and witnesses_r1 16:48:32; E0 16:50:39; baseline gate 16:51:34; independent code audit 16:57:49; first focal mask0 16:58:00; result revision manifest 16:58:21. Per-mask raw outputs are retained. Statistical per-null outputs do not apply to this exact finite-source rank calculation.

## Permitted classification

Exploratory result, qualified as an exact finite-source capacity test under declared localized charge and ideal pulse-family observer assumptions. This closes mixture and same-mask fingerprint readability in the specified finite model. It does not establish source specificity, biological preparation, physiological implementation, EEG realization, consciousness or the empirical identification of three subjectivities.
