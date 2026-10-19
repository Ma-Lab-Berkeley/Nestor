/** Display equations of the page, in the paper's notation (macros in components/Math.tsx). */
export const eq = {
  moe: String.raw`\vx_{j} + \sum_{e \in \cE_{j}} \alpha_{je}\, \MLP_{e}(\vx_{j}),
    \qquad \cE_{j} \doteq \TopK\big(\vr(\vx_{j}),\, \Ke\big) \subseteq [E]`,

  columnAttention: String.raw`\va_{j} \doteq \vx_{j} + \sum_{c \in \cC_{j}} w_{jc}\,
    \Attn_{c}\big(\vx_{j},\, (\vx_{i})_{i \in \cU_{c},\, i \leq j}\big),
    \qquad \cC_{j} \doteq \TopK\big(\vs(\vx_{j}),\, \Kc\big) \subseteq [C]`,

  nested: String.raw`\vx_{j}^{+} \doteq \va_{j} + \sum_{c \in \cC_{j}} w_{jc}
    \sum_{e \in \cE_{jc}} \alpha_{jce}\, \MLP_{c,e}(\va_{j}),
    \qquad \cE_{jc} \doteq \TopK\big(\vr_{c}(\va_{j}),\, \Ke\big)`,

  routerGate: String.raw`\cC_{j} \doteq \TopK(\vs_{j} + \vb,\, \Kc) \subseteq [C],
    \qquad w_{jc} \doteq \frac{\exp(s_{jc})}{\sum_{c' \in \cC_{j}} \exp(s_{jc'})} \quad \text{for } c \in \cC_{j}`,

  routerBias: String.raw`b_{c} \leftarrow b_{c} + \gamma\, \sign\Big(\frac{1}{C} \sum_{c' \in [C]} n_{c'} - n_{c}\Big),
    \qquad \gamma = 10^{-3}`,

  attnRes: String.raw`\AttnRes(\vz_{1}, \dots, \vz_{N}) \doteq \sum_{i=1}^{N}
    \frac{\exp\big(\ip{\vq}{\vz_{i}} / (\sqrt{D}\, \rms(\vz_{i}))\big)}
         {\sum_{i'=1}^{N} \exp\big(\ip{\vq}{\vz_{i'}} / (\sqrt{D}\, \rms(\vz_{i'}))\big)}\;
    g_{i}\, \frac{\vz_{i}}{\rms(\vz_{i})}`,

  tokensMultiplier: String.raw`R(S) = \frac{S_{0} + S}{S_{0} + \rho\, S}
    \;\xrightarrow{\;S \to \infty\;}\; \frac{1}{\rho}`,

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

  dims: String.raw`\dN \approx C\,(D^{2} + E r D), \qquad \dF \approx C\,(C D^{2} + 2 E r D),
    \qquad \frac{\dF}{\dN} \approx \frac{C D + 2 E r}{D + E r} \in [2, C]`,

  multiplier: String.raw`\frac{\dF}{\dN} \cdot \frac{S_{0} + S - 1}{S_{0} + (S - 1) / C},
    \qquad S_{0} \doteq F_{0} / F_{1}`,

  forgetting: String.raw`\frac{t^{2}}{2}\, \norm{\vB_{ce}^{\star} \vA_{c}^{\star} \vA_{c_{Q}}^{\star\top}\, \vXi\, \vSigma_{ce}^{1/2}}_{F}^{2} + O(t^{3}),
    \qquad \vSigma_{ce} \doteq \Ex[\vm \vm^{\top} \mid c, e]`,
} as const;
