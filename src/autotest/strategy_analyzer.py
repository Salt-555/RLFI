import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from datetime import datetime
import os
import json

from .strategy_database import StrategyDatabase


class StrategyAnalyzer:
    """
    Analyzes and compares trading strategies across weeks to identify patterns
    """
    
    def __init__(self, db_path: str = 'autotest_strategies.db'):
        self.db = StrategyDatabase(db_path)
    
    def generate_weekly_report(self, weeks_back: int = 4) -> Dict[str, Any]:
        """
        Generate comprehensive weekly comparison report
        
        Args:
            weeks_back: Number of weeks to analyze
        
        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*80}")
        print(f"STRATEGY ANALYSIS - LAST {weeks_back} WEEKS")
        print(f"{'='*80}\n")
        
        # Get weekly summaries
        weekly_summaries = self.db.get_weekly_summaries(weeks_back)
        
        if weekly_summaries.empty:
            return {'error': 'No historical data available'}
        
        # Get top strategies
        top_strategies = self.db.get_top_strategies(weeks_back, top_n=20)
        
        # Analyze parameter patterns
        patterns = self.db.analyze_parameter_patterns()
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'weeks_analyzed': len(weekly_summaries),
            'total_strategies_tested': len(top_strategies),
            'weekly_performance': self._analyze_weekly_performance(weekly_summaries),
            'best_parameters': self._identify_best_parameters(patterns),
            'top_strategies': self._format_top_strategies(top_strategies),
            'recommendations': self._generate_recommendations(patterns, weekly_summaries)
        }
        
        self._print_report(report)
        
        return report
    
    def _analyze_weekly_performance(self, weekly_summaries: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance trends across weeks"""
        if weekly_summaries.empty:
            return {}
        
        return {
            'avg_sharpe_ratio': float(weekly_summaries['best_sharpe_ratio'].mean()),
            'avg_total_return': float(weekly_summaries['best_total_return'].mean()),
            'best_week': {
                'week_number': int(weekly_summaries.loc[weekly_summaries['best_sharpe_ratio'].idxmax(), 'week_number']),
                'sharpe_ratio': float(weekly_summaries['best_sharpe_ratio'].max()),
                'total_return': float(weekly_summaries['best_total_return'].max())
            },
            'trend': 'improving' if weekly_summaries['best_sharpe_ratio'].iloc[-1] > weekly_summaries['best_sharpe_ratio'].iloc[0] else 'declining'
        }
    
    def _identify_best_parameters(self, patterns: pd.DataFrame) -> Dict[str, Any]:
        """Identify which parameters work best"""
        if patterns.empty:
            return {}
        
        best_params = {}
        
        # Best learning rates
        lr_patterns = patterns[patterns['pattern_type'] == 'learning_rate']
        if not lr_patterns.empty:
            best_lr = lr_patterns.loc[lr_patterns['avg_sharpe_ratio'].idxmax()]
            best_params['learning_rate'] = {
                'value': float(best_lr['pattern_value']),
                'avg_sharpe': float(best_lr['avg_sharpe_ratio']),
                'success_rate': float(best_lr['success_rate'])
            }
        
        # Best ticker combinations
        ticker_patterns = patterns[patterns['pattern_type'] == 'tickers']
        if not ticker_patterns.empty:
            best_tickers = ticker_patterns.nlargest(5, 'avg_sharpe_ratio')
            best_params['top_ticker_combinations'] = [
                {
                    'tickers': json.loads(row['pattern_value']),
                    'avg_sharpe': float(row['avg_sharpe_ratio']),
                    'success_rate': float(row['success_rate'])
                }
                for _, row in best_tickers.iterrows()
            ]
        
        return best_params
    
    def _format_top_strategies(self, top_strategies: pd.DataFrame) -> List[Dict[str, Any]]:
        """Format top strategies for report"""
        if top_strategies.empty:
            return []
        
        strategies = []
        for _, row in top_strategies.head(10).iterrows():
            strategies.append({
                'model_id': row['model_id'],
                'tickers': json.loads(row['tickers']),
                'learning_rate': float(row['learning_rate']) if pd.notna(row['learning_rate']) else None,
                'sharpe_ratio': float(row['sharpe_ratio']) if pd.notna(row['sharpe_ratio']) else None,
                'total_return': float(row['total_return']) if pd.notna(row['total_return']) else None,
                'week_number': int(row['week_number'])
            })
        
        return strategies
    
    def _generate_recommendations(self, patterns: pd.DataFrame, 
                                 weekly_summaries: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if not patterns.empty:
            # Learning rate recommendation
            lr_patterns = patterns[patterns['pattern_type'] == 'learning_rate']
            if not lr_patterns.empty:
                best_lr = lr_patterns.loc[lr_patterns['avg_sharpe_ratio'].idxmax()]
                recommendations.append(
                    f"Focus on learning rate {best_lr['pattern_value']} "
                    f"(avg Sharpe: {best_lr['avg_sharpe_ratio']:.3f})"
                )
            
            # Ticker recommendation
            ticker_patterns = patterns[patterns['pattern_type'] == 'tickers']
            if not ticker_patterns.empty:
                best_tickers = ticker_patterns.loc[ticker_patterns['avg_sharpe_ratio'].idxmax()]
                recommendations.append(
                    f"Best ticker combination: {best_tickers['pattern_value']} "
                    f"(success rate: {best_tickers['success_rate']*100:.1f}%)"
                )
        
        if not weekly_summaries.empty and len(weekly_summaries) >= 2:
            recent_trend = weekly_summaries['best_sharpe_ratio'].iloc[-3:].mean()
            older_trend = weekly_summaries['best_sharpe_ratio'].iloc[:3].mean()
            
            if recent_trend > older_trend:
                recommendations.append("Performance is improving - continue current parameter exploration")
            else:
                recommendations.append("Performance declining - consider adjusting parameter ranges")
        
        return recommendations
    
    def _print_report(self, report: Dict[str, Any]):
        """Print formatted report to console"""
        print("\n" + "="*80)
        print("WEEKLY PERFORMANCE SUMMARY")
        print("="*80)
        
        if 'weekly_performance' in report and report['weekly_performance']:
            perf = report['weekly_performance']
            print(f"\nAverage Best Sharpe Ratio: {perf['avg_sharpe_ratio']:.3f}")
            print(f"Average Best Return: {perf['avg_total_return']*100:.2f}%")
            print(f"Performance Trend: {perf['trend'].upper()}")
            
            if 'best_week' in perf:
                print(f"\nBest Week: #{perf['best_week']['week_number']}")
                print(f"  Sharpe: {perf['best_week']['sharpe_ratio']:.3f}")
                print(f"  Return: {perf['best_week']['total_return']*100:.2f}%")
        
        print("\n" + "="*80)
        print("BEST PARAMETERS IDENTIFIED")
        print("="*80)
        
        if 'best_parameters' in report and report['best_parameters']:
            params = report['best_parameters']
            
            if 'learning_rate' in params:
                lr = params['learning_rate']
                print(f"\nBest Learning Rate: {lr['value']}")
                print(f"  Avg Sharpe: {lr['avg_sharpe']:.3f}")
                print(f"  Success Rate: {lr['success_rate']*100:.1f}%")
            
            if 'top_ticker_combinations' in params:
                print("\nTop Ticker Combinations:")
                for i, combo in enumerate(params['top_ticker_combinations'][:5], 1):
                    print(f"  {i}. {combo['tickers']}")
                    print(f"     Sharpe: {combo['avg_sharpe']:.3f}, Success: {combo['success_rate']*100:.1f}%")
        
        print("\n" + "="*80)
        print("TOP 10 STRATEGIES ALL-TIME")
        print("="*80)
        
        if 'top_strategies' in report:
            for i, strat in enumerate(report['top_strategies'][:10], 1):
                print(f"\n{i}. {strat['model_id']}")
                print(f"   Tickers: {strat['tickers']}")
                print(f"   Sharpe: {strat['sharpe_ratio']:.3f}, Return: {strat['total_return']*100:.2f}%")
                print(f"   Learning Rate: {strat['learning_rate']}")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if 'recommendations' in report:
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print("\n" + "="*80)
    
    def compare_strategies(self, model_id1: str, model_id2: str) -> Dict[str, Any]:
        """Compare two specific strategies"""
        query = '''
            SELECT s.*, b.sharpe_ratio, b.total_return, b.max_drawdown, b.win_rate
            FROM strategies s
            JOIN backtest_results b ON s.model_id = b.model_id
            WHERE s.model_id IN (?, ?)
        '''
        df = pd.read_sql_query(query, self.db.conn, params=(model_id1, model_id2))
        
        if len(df) < 2:
            return {'error': 'One or both strategies not found'}
        
        strat1 = df[df['model_id'] == model_id1].iloc[0]
        strat2 = df[df['model_id'] == model_id2].iloc[0]
        
        return {
            'strategy1': {
                'model_id': strat1['model_id'],
                'tickers': json.loads(strat1['tickers']),
                'learning_rate': float(strat1['learning_rate']),
                'sharpe_ratio': float(strat1['sharpe_ratio']),
                'total_return': float(strat1['total_return'])
            },
            'strategy2': {
                'model_id': strat2['model_id'],
                'tickers': json.loads(strat2['tickers']),
                'learning_rate': float(strat2['learning_rate']),
                'sharpe_ratio': float(strat2['sharpe_ratio']),
                'total_return': float(strat2['total_return'])
            },
            'comparison': {
                'sharpe_diff': float(strat2['sharpe_ratio'] - strat1['sharpe_ratio']),
                'return_diff': float(strat2['total_return'] - strat1['total_return']),
                'better_strategy': model_id2 if strat2['sharpe_ratio'] > strat1['sharpe_ratio'] else model_id1
            }
        }
    
    def visualize_performance_trends(self, output_dir: str = 'autotest_results', 
                                    weeks_back: int = 12):
        """Create visualization charts for strategy performance"""
        os.makedirs(output_dir, exist_ok=True)
        
        weekly_summaries = self.db.get_weekly_summaries(weeks_back)
        
        if weekly_summaries.empty:
            print("No data available for visualization")
            return
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Sharpe Ratio Trend
        axes[0, 0].plot(weekly_summaries['week_number'], 
                       weekly_summaries['best_sharpe_ratio'], 
                       marker='o', linewidth=2, markersize=8)
        axes[0, 0].set_title('Best Sharpe Ratio Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Week Number')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Total Return Trend
        axes[0, 1].plot(weekly_summaries['week_number'], 
                       weekly_summaries['best_total_return'] * 100, 
                       marker='s', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_title('Best Total Return Over Time', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Week Number')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Models Trained vs Performance
        axes[1, 0].scatter(weekly_summaries['models_trained'], 
                          weekly_summaries['best_sharpe_ratio'],
                          s=100, alpha=0.6)
        axes[1, 0].set_title('Models Trained vs Best Sharpe', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Number of Models Trained')
        axes[1, 0].set_ylabel('Best Sharpe Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Average vs Best Performance
        axes[1, 1].plot(weekly_summaries['week_number'], 
                       weekly_summaries['avg_sharpe_ratio'], 
                       marker='o', label='Average', linewidth=2)
        axes[1, 1].plot(weekly_summaries['week_number'], 
                       weekly_summaries['best_sharpe_ratio'], 
                       marker='s', label='Best', linewidth=2)
        axes[1, 1].set_title('Average vs Best Sharpe Ratio', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Week Number')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'strategy_trends_{datetime.now().strftime("%Y%m%d")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
        plt.close()
        
        # Create parameter analysis chart
        self._visualize_parameter_patterns(output_dir)
    
    def _visualize_parameter_patterns(self, output_dir: str):
        """Visualize which parameters work best"""
        patterns = self.db.analyze_parameter_patterns()
        
        if patterns.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Learning rate performance
        lr_patterns = patterns[patterns['pattern_type'] == 'learning_rate'].copy()
        if not lr_patterns.empty:
            lr_patterns['pattern_value'] = lr_patterns['pattern_value'].astype(float)
            lr_patterns = lr_patterns.sort_values('pattern_value')
            
            axes[0].bar(range(len(lr_patterns)), lr_patterns['avg_sharpe_ratio'])
            axes[0].set_xticks(range(len(lr_patterns)))
            axes[0].set_xticklabels([f"{x:.4f}" for x in lr_patterns['pattern_value']], rotation=45)
            axes[0].set_title('Learning Rate vs Avg Sharpe Ratio', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Learning Rate')
            axes[0].set_ylabel('Avg Sharpe Ratio')
            axes[0].grid(True, alpha=0.3, axis='y')
        
        # Ticker combination performance
        ticker_patterns = patterns[patterns['pattern_type'] == 'tickers'].nlargest(10, 'avg_sharpe_ratio')
        if not ticker_patterns.empty:
            ticker_labels = [json.loads(t)[:2] for t in ticker_patterns['pattern_value']]  # First 2 tickers
            ticker_labels = [f"{t[0]},{t[1]}..." for t in ticker_labels]
            
            axes[1].barh(range(len(ticker_patterns)), ticker_patterns['avg_sharpe_ratio'])
            axes[1].set_yticks(range(len(ticker_patterns)))
            axes[1].set_yticklabels(ticker_labels)
            axes[1].set_title('Top Ticker Combinations', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Avg Sharpe Ratio')
            axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'parameter_analysis_{datetime.now().strftime("%Y%m%d")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Parameter analysis saved to: {output_path}")
        plt.close()
    
    def export_analysis_report(self, output_dir: str = 'autotest_results', 
                              weeks_back: int = 12):
        """Export comprehensive analysis report"""
        report = self.generate_weekly_report(weeks_back)
        
        # Save as JSON
        report_path = os.path.join(output_dir, f'strategy_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nAnalysis report saved to: {report_path}")
        
        # Export database to CSV
        self.db.export_to_csv(output_dir)
        
        # Create visualizations
        self.visualize_performance_trends(output_dir, weeks_back)
        
        return report_path
