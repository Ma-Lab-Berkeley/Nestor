import { rungs } from "../content/paper";

/** The twelve pretraining runs: Table 2 of the paper. */
export function ResultsTable() {
  return (
    <div className="table-wrap">
      <table className="results">
        <thead>
          <tr>
            <th scope="col">Size</th>
            <th scope="col">Tokens</th>
            <th scope="col">
              Compute
              <span className="th-sub">PFLOP/s-days</span>
            </th>
            <th scope="col">Model</th>
            <th scope="col" className="num">
              Total params
            </th>
            <th scope="col" className="num">
              Held-out loss
            </th>
            <th scope="col" className="num">
              Compute at equal loss
            </th>
          </tr>
        </thead>
        <tbody>
          {rungs.map((r) =>
            r.rows.map((row, i) => (
              <tr key={`${r.size}-${row.model}`} className={row.model === "Nestor" ? "is-ours" : undefined}>
                {i === 0 && (
                  <>
                    <th scope="rowgroup" rowSpan={3}>
                      {r.size}
                    </th>
                    <td rowSpan={3}>{r.tokens}</td>
                    <td rowSpan={3}>{r.compute}</td>
                  </>
                )}
                <td>{row.model}</td>
                <td className="num">{row.params}</td>
                <td className="num">{row.loss}</td>
                <td className="num">{row.multiplier}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
    </div>
  );
}
