import statistics

class PerformanceDashboard:
    def __init__(self):
        self.daily_results = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.pnl_list = []

    def record_trade(self, trade_result):
        self.daily_results.append(trade_result)
        self.total_trades += 1
        pnl = trade_result.get("pnl", 0)
        self.pnl_list.append(pnl)
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

    def generate_summary(self):
        if not self.daily_results:
            return "📊 No trades recorded today."

        total_pnl = sum(self.pnl_list)
        avg_pnl = statistics.mean(self.pnl_list) if self.pnl_list else 0
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        best_trade = max(self.pnl_list) if self.pnl_list else 0
        worst_trade = min(self.pnl_list) if self.pnl_list else 0

        summary = (
            f"<b>📈 Daily Performance Dashboard</b>\n\n"
            f"🔸 Trades: {self.total_trades}\n"
            f"✅ Wins: {self.winning_trades}, ❌ Losses: {self.losing_trades}\n"
            f"💰 Net PnL: {total_pnl:.2f}%\n"
            f"📉 Avg PnL: {avg_pnl:.2f}%\n"
            f"🏆 Best Trade: {best_trade:.2f}%\n"
            f"⚠️ Worst Trade: {worst_trade:.2f}%\n"
            f"📊 Win Rate: {win_rate:.2f}%"
        )

        return summary
