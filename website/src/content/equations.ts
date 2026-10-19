/** Display equations of the page, in the paper's notation (macros in components/Math.tsx). */
export const eq = {
  linearModel: String.raw`\vm \doteq \sum_{i \in \cI} \big(\vz_{S}^{\top} \vW_{c}^{\star} \vz_{i}\big)\, \vz_{i},
    \qquad \cI \doteq \{\, i < S : c_{i} = c \,\},
    \qquad \vy \doteq \vB_{ce}^{\star} \vA_{c}^{\star} \vm + \veps`,

  nestedStructure: String.raw`\vG_{ce}^{\star} \doteq \vB_{ce}^{\star} \vA_{c}^{\star},
    \qquad \rank \vG_{ce}^{\star} = r,
    \qquad \row(\vG_{ce}^{\star}) = \row(\vA_{c}^{\star}) \ \text{ for all } e \in [E]`,

  predictors: String.raw`\begin{aligned}
    \vf^{\mathrm{N}}(c, e, \vX) &\doteq \vB_{ce} \vA_{c} \sum_{i \in \cI} (\vz_{S}^{\top} \vW_{c} \vz_{i})\, \vz_{i}, \\
    \vf^{\mathrm{F}}(c, e, \vX) &\doteq \vW_{ce}^{\mathrm{out}} \vW_{ce}^{\mathrm{in}} \vV \sum_{i < S} (\vz_{S}^{\top} \vW_{c c_{i}} \vz_{i})\, \vz_{i}.
    \end{aligned}`,

  leastSquares: String.raw`\hat{\vf} \in \argmin_{\vf \in \cM} \sum_{i \in [T]} \norm{\vy_{i} - \vf(c_{i}, e_{i}, \vX_{i})}_{2}^{2},
    \qquad \cE(\hat{\vf}) \doteq \Ex\big[\norm{\hat{\vf}(c, e, \vX) - \vf^{\star}(c, e, \vX)}_{2}^{2} \,\big|\, \hat{\vf}\big]`,

  dims: String.raw`\dN \approx C\,(D^{2} + E r D), \qquad \dF \approx C\,(C D^{2} + 2 E r D),
    \qquad \frac{\dF}{\dN} \approx \frac{C D + 2 E r}{D + E r} \in [2, C]`,

  multiplier: String.raw`\frac{\dF}{\dN} \cdot \frac{S_{0} + S - 1}{S_{0} + (S - 1) / C},
    \qquad S_{0} \doteq F_{0} / F_{1}`,

  forgetting: String.raw`\frac{t^{2}}{2}\, \norm{\vB_{ce}^{\star} \vA_{c}^{\star} \vA_{c_{Q}}^{\star\top}\, \vXi\, \vSigma_{ce}^{1/2}}_{F}^{2} + O(t^{3}),
    \qquad \vSigma_{ce} \doteq \Ex[\vm \vm^{\top} \mid c, e]`,
} as const;
