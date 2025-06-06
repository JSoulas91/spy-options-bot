import statistics, numpy as np

class PerformanceDashboard:
    def __init__(self):
        self.daily_results = []
        self.pnl_list      = []
        self.latencies     = []   # ms between signal and fill
        self.total_trades  = self.winning = self.losing = 0

    def record_trade(self, trade):
        pnl = trade.get("pnl", 0)
        latency = trade.get("latency_ms", None)

        self.daily_results.append(trade)
        self.pnl_list.append(pnl)
        self.total_trades += 1
        self.winning += pnl > 0
        self.losing  += pnl <= 0
        if latency is not None:
            self.latencies.append(latency)

    # ── Sharpe calc (daily) ──
    @staticmethod
    def _sharpe(pnls):
        if len(pnls) < 2: return 0.0
        r = np.array(pnls)
        return (r.mean() / (r.std() + 1e-9)) * np.sqrt(len(r))

    def generate_summary(self):
        if not self.pnl_list:
            return "📊 No trades recorded today."

        total = sum(self.pnl_list)
        avg   = statistics.mean(self.pnl_list)
        winrt = (self.winning / self.total_trades) * 100
        sharpe= self._sharpe(self.pnl_list)
        best, worst = max(self.pnl_list), min(self.pnl_list)
        avg_lat = statistics.mean(self.latencies) if self.latencies else "N/A"

        return (
            f"<b>📈 Daily Performance</b>\n\n"
            f"Trades: {self.total_trades}\n"
            f"✅ Wins: {self.winning} | ❌ Losses: {self.losing}\n"
            f"Net PnL: {total:.2f}% | Avg: {avg:.2f}%\n"
            f"🏆 Best: {best:.2f}% | ⚠️ Worst: {worst:.2f}%\n"
            f"📊 Win Rate: {winrt:.1f}%\n"
            f"⚖️ Sharpe*: {sharpe:.2f}\n"
            f"⏱️ Avg Latency: {avg_lat}"
        )