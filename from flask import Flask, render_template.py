from flask import Flask, render_template, jsonify, request
import sqlite3
import json
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils

app = Flask(__name__)

class DashboardManager:
    def __init__(self, db_path="review_analysis.db"):
        self.db_path = db_path
    
    def get_app_list(self):
        """Get list of all apps in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT app_id, COUNT(*) as review_count, MAX(created_at) as last_updated
            FROM reviews 
            GROUP BY app_id 
            ORDER BY last_updated DESC
        """)
        
        apps = cursor.fetchall()
        conn.close()
        
        return [{"app_id": app[0], "review_count": app[1], "last_updated": app[2]} for app in apps]
    
    def get_app_summary(self, app_id, days=30):
        """Get summary statistics for an app"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get overall stats
        cursor.execute("""
            SELECT 
                sentiment,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                platform
            FROM reviews 
            WHERE app_id = ? AND date >= datetime('now', '-{} days')
            GROUP BY sentiment, platform
        """.format(days), (app_id,))
        
        platform_stats = cursor.fetchall()
        
        # Get trend data
        cursor.execute("""
            SELECT 
                DATE(created_at) as date,
                sentiment,
                COUNT(*) as count
            FROM reviews 
            WHERE app_id = ? AND created_at >= datetime('now', '-{} days')
            GROUP BY DATE(created_at), sentiment
            ORDER BY date
        """.format(days), (app_id,))
        
        trend_data = cursor.fetchall()
        
        # Get recent reviews
        cursor.execute("""
            SELECT content, sentiment, confidence, platform, rating, date
            FROM reviews 
            WHERE app_id = ? 
            ORDER BY created_at DESC 
            LIMIT 10
        """, (app_id,))
        
        recent_reviews = cursor.fetchall()
        
        conn.close()
        
        return {
            "platform_stats": platform_stats,
            "trend_data": trend_data,
            "recent_reviews": recent_reviews
        }
    
    def get_dashboard_data(self, app_id=None, days=30):
        """Get comprehensive dashboard data"""
        if app_id:
            return self.get_app_summary(app_id, days)
        else:
            return {"apps": self.get_app_list()}
    
    def create_sentiment_chart(self, app_id, days=30):
        """Create sentiment distribution chart"""
        data = self.get_app_summary(app_id, days)
        
        # Aggregate sentiment counts across platforms
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0, "unknown": 0}
        
        for stat in data["platform_stats"]:
            sentiment, count, confidence, platform = stat
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += count
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(sentiment_counts.keys()),
            values=list(sentiment_counts.values()),
            hole=0.4,
            marker_colors=['#28a745', '#dc3545', '#ffc107', '#6c757d']
        )])
        
        fig.update_layout(
            title=f"Sentiment Distribution - {app_id}",
            font=dict(size=16),
            showlegend=True
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def create_trend_chart(self, app_id, days=30):
        """Create trend analysis chart"""
        data = self.get_app_summary(app_id, days)
        
        # Process trend data
        dates = []
        positive_counts = []
        negative_counts = []
        neutral_counts = []
        
        trend_dict = {}
        for date, sentiment, count in data["trend_data"]:
            if date not in trend_dict:
                trend_dict[date] = {"positive": 0, "negative": 0, "neutral": 0}
            trend_dict[date][sentiment] = count
        
        for date in sorted(trend_dict.keys()):
            dates.append(date)
            positive_counts.append(trend_dict[date]["positive"])
            negative_counts.append(trend_dict[date]["negative"])
            neutral_counts.append(trend_dict[date]["neutral"])
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=positive_counts,
            mode='lines+markers',
            name='Positive',
            line=dict(color='#28a745')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=negative_counts,
            mode='lines+markers',
            name='Negative',
            line=dict(color='#dc3545')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=neutral_counts,
            mode='lines+markers',
            name='Neutral',
            line=dict(color='#ffc107')
        ))
        
        fig.update_layout(
            title=f"Sentiment Trends - {app_id}",
            xaxis_title="Date",
            yaxis_title="Review Count",
            font=dict(size=14),
            hovermode='x unified'
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

dashboard_manager = DashboardManager()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/apps')
def get_apps():
    """API endpoint to get list of apps"""
    apps = dashboard_manager.get_app_list()
    return jsonify(apps)

@app.route('/api/app/<app_id>')
def get_app_data(app_id):
    """API endpoint to get app-specific data"""
    days = request.args.get('days', 30, type=int)
    data = dashboard_manager.get_app_summary(app_id, days)
    return jsonify(data)

@app.route('/api/chart/sentiment/<app_id>')
def get_sentiment_chart(app_id):
    """API endpoint to get sentiment chart data"""
    days = request.args.get('days', 30, type=int)
    chart_data = dashboard_manager.create_sentiment_chart(app_id, days)
    return chart_data

@app.route('/api/chart/trend/<app_id>')
def get_trend_chart(app_id):
    """API endpoint to get trend chart data"""
    days = request.args.get('days', 30, type=int)
    chart_data = dashboard_manager.create_trend_chart(app_id, days)
    return chart_data

# Template for the HTML dashboard
dashboard_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>App Review Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Arial', sans-serif;
        }
        
        .dashboard-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin: 20px;
            padding: 30px;
        }
        
        .card {
            border: none;
            border-radius: 15px;
            background: linear-gradient(145deg, #ffffff, #f0f0f0);
            box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .stat-card {
            text-align: center;
            padding: 20px;
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .app-selector {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 20px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .app-selector:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
        }
        
        .refresh-btn {
            background: linear-gradient(45deg, #28a745, #20c997);
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .refresh-btn:hover {
            transform: scale(1.1);
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 50px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .recent-reviews {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .review-item {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }
        
        .sentiment-badge {
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .sentiment-positive { background: #d4edda; color: #155724; }
        .sentiment-negative { background: #f8d7da; color: #721c24; }
        .sentiment-neutral { background: #fff3cd; color: #856404; }
        .sentiment-unknown { background: #d1ecf1; color: #0c5460; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="dashboard-container">
            <div class="row mb-4">
                <div class="col-12">
                    <h1 class="text-center mb-4">
                        <i class="fas fa-chart-line"></i> App Review Analysis Dashboard
                    </h1>
                    <div class="d-flex justify-content-center align-items-center gap-3">
                        <select id="appSelector" class="app-selector">
                            <option value="">Select an App</option>
                        </select>
                        <select id="daysSelector" class="app-selector">
                            <option value="7">Last 7 days</option>
                            <option value="30" selected>Last 30 days</option>
                            <option value="90">Last 90 days</option>
                        </select>
                        <button id="refreshBtn" class="refresh-btn">
                            <i class="fas fa-sync-alt"></i> Refresh
                        </button>
                    </div>
                </div>
            </div>
            
            <div id="loadingSpinner" class="loading">
                <div class="spinner"></div>
                <p>Loading data...</p>
            </div>
            
            <div id="dashboardContent" style="display: none;">
                <!-- Statistics Cards -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card stat-card">
                            <div class="card-body">
                                <i class="fas fa-comments fa-2x text-primary mb-2"></i>
                                <div class="stat-number" id="totalReviews">0</div>
                                <h6>Total Reviews</h6>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card stat-card">
                            <div class="card-body">
                                <i class="fas fa-thumbs-up fa-2x text-success mb-2"></i>
                                <div class="stat-number" id="positiveReviews">0</div>
                                <h6>Positive Reviews</h6>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card stat-card">
                            <div class="card-body">
                                <i class="fas fa-thumbs-down fa-2x text-danger mb-2"></i>
                                <div class="stat-number" id="negativeReviews">0</div>
                                <h6>Negative Reviews</h6>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card stat-card">
                            <div class="card-body">
                                <i class="fas fa-minus fa-2x text-warning mb-2"></i>
                                <div class="stat-number" id="neutralReviews">0</div>
                                <h6>Neutral Reviews</h6>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Charts -->
                <div class="row">
                    <div class="col-md-6">
                        <div class="chart-container">
                            <div id="sentimentChart"></div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="chart-container">
                            <div id="trendChart"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Recent Reviews -->
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h5><i class="fas fa-clock"></i> Recent Reviews</h5>
                            </div>
                            <div class="card-body">
                                <div id="recentReviews" class="recent-reviews"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        class Dashboard {
            constructor() {
                this.currentApp = null;
                this.currentDays = 30;
                this.init();
            }
            
            init() {
                this.loadApps();
                this.setupEventListeners();
            }
            
            setupEventListeners() {
                document.getElementById('appSelector').addEventListener('change', (e) => {
                    this.currentApp = e.target.value;
                    if (this.currentApp) {
                        this.loadAppData();
                    }
                });
                
                document.getElementById('daysSelector').addEventListener('change', (e) => {
                    this.currentDays = parseInt(e.target.value);
                    if (this.currentApp) {
                        this.loadAppData();
                    }
                });
                
                document.getElementById('refreshBtn').addEventListener('click', () => {
                    if (this.currentApp) {
                        this.loadAppData();
                    } else {
                        this.loadApps();
                    }
                });
            }
            
            showLoading() {
                document.getElementById('loadingSpinner').style.display = 'block';
                document.getElementById('dashboardContent').style.display = 'none';
            }
            
            hideLoading() {
                document.getElementById('loadingSpinner').style.display = 'none';
                document.getElementById('dashboardContent').style.display = 'block';
            }
            
            async loadApps() {
                try {
                    const response = await fetch('/api/apps');
                    const apps = await response.json();
                    
                    const selector = document.getElementById('appSelector');
                    selector.innerHTML = '<option value="">Select an App</option>';
                    
                    apps.forEach(app => {
                        const option = document.createElement('option');
                        option.value = app.app_id;
                        option.textContent = `${app.app_id} (${app.review_count} reviews)`;
                        selector.appendChild(option);
                    });
                } catch (error) {
                    console.error('Error loading apps:', error);
                }
            }
            
            async loadAppData() {
                if (!this.currentApp) return;
                
                this.showLoading();
                
                try {
                    const response = await fetch(`/api/app/${this.currentApp}?days=${this.currentDays}`);
                    const data = await response.json();
                    
                    this.updateStatistics(data);
                    await this.loadCharts();
                    this.updateRecentReviews(data.recent_reviews);
                    
                    this.hideLoading();
                } catch (error) {
                    console.error('Error loading app data:', error);
                    this.hideLoading();
                }
            }
            
            updateStatistics(data) {
                // Aggregate statistics across platforms
                const stats = { positive: 0, negative: 0, neutral: 0, unknown: 0, total: 0 };
                
                data.platform_stats.forEach(stat => {
                    const [sentiment, count] = stat;
                    if (stats.hasOwnProperty(sentiment)) {
                        stats[sentiment] += count;
                        stats.total += count;
                    }
                });
                
                document.getElementById('totalReviews').textContent = stats.total;
                document.getElementById('positiveReviews').textContent = stats.positive;
                document.getElementById('negativeReviews').textContent = stats.negative;
                document.getElementById('neutralReviews').textContent = stats.neutral;
            }
            
            async loadCharts() {
                try {
                    // Load sentiment chart
                    const sentimentResponse = await fetch(`/api/chart/sentiment/${this.currentApp}?days=${this.currentDays}`);
                    const sentimentData = await sentimentResponse.json();
                    Plotly.newPlot('sentimentChart', sentimentData.data, sentimentData.layout);
                    
                    // Load trend chart
                    const trendResponse = await fetch(`/api/chart/trend/${this.currentApp}?days=${this.currentDays}`);
                    const trendData = await trendResponse.json();
                    Plotly.newPlot('trendChart', trendData.data, trendData.layout);
                } catch (error) {
                    console.error('Error loading charts:', error);
                }
            }
            
            updateRecentReviews(reviews) {
                const container = document.getElementById('recentReviews');
                container.innerHTML = '';
                
                if (!reviews || reviews.length === 0) {
                    container.innerHTML = '<p class="text-muted">No recent reviews found.</p>';
                    return;
                }
                
                reviews.forEach(review => {
                    const [content, sentiment, confidence, platform, rating, date] = review;
                    
                    const reviewDiv = document.createElement('div');
                    reviewDiv.className = 'review-item';
                    
                    reviewDiv.innerHTML = `
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <span class="sentiment-badge sentiment-${sentiment}">${sentiment.toUpperCase()}</span>
                            <div class="text-muted small">
                                <i class="fas fa-star"></i> ${rating}/5 | ${platform} | ${new Date(date).toLocaleDateString()}
                            </div>
                        </div>
                        <p class="mb-1">${content.substring(0, 200)}${content.length > 200 ? '...' : ''}</p>
                        <small class="text-muted">Confidence: ${(confidence * 100).toFixed(1)}%</small>
                    `;
                    
                    container.appendChild(reviewDiv);
                });
            }
        }
        
        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new Dashboard();
        });
    </script>
</body>
</html>
'''

# Create templates directory and save HTML
import os
if not os.path.exists('templates'):
    os.makedirs('templates')

with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(dashboard_html)

if __name__ == '__main__':
    print("Starting Review Analysis Dashboard...")
    print("Dashboard will be available at: http://localhost:5000")
    print("\nFeatures:")
    print("- Real-time sentiment analysis visualization")
    print("- Multi-platform review aggregation")
    print("- Trend analysis over time")
    print("- Recent reviews display")
    print("- Responsive design with modern UI")
    
    app.run(debug=True, host='0.0.0.0', port=5000)