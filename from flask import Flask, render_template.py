import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, TrendingDown, Minus, AlertTriangle, CheckCircle, Clock, Star, Filter, RefreshCw } from 'lucide-react';

const ReviewAnalysisDashboard = () => {
  const [selectedApp, setSelectedApp] = useState('');
  const [analysisData, setAnalysisData] = useState(null);
  const [trendData, setTrendData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  // Mock data - Replace with actual API calls
  const mockAnalysisData = {
    'com.example.app1': {
      app_id: 'com.example.app1',
      app_name: 'Sample App 1',
      analysis_date: '2024-01-15T10:30:00Z',
      total_reviews_analyzed: 847,
      total_reviews_filtered: 153,
      sentiment_distribution: {
        positive: 68.5,
        negative: 18.2,
        neutral: 13.3
      },
      overall_sentiment: 'positive',
      confidence_metrics: {
        average_confidence: 0.78,
        high_confidence_reviews: 612,
        low_confidence_reviews: 89
      },
      quality_metrics: {
        average_quality_score: 0.72,
        high_quality_reviews: 645,
        low_quality_reviews: 47
      },
      platform_analysis: {
        app_store_weight: 0.65,
        play_store_weight: 0.35,
        app_store_reviews: 551,
        play_store_reviews: 296
      },
      trend_analysis: {
        trend_direction: 'improving',
        anomaly_detected: false,
        confidence_change: 0.05,
        weekly_sentiment_change: {
          positive: 0.08,
          negative: -0.05,
          neutral: -0.03
        }
      },
      flags: {
        manual_review_needed: false,
        anomaly_detected: false,
        low_confidence_batch: false
      },
      language_distribution: {
        'en': 523,
        'es': 189,
        'zh': 85,
        'hi': 34,
        'ar': 16
      }
    },
    'com.example.app2': {
      app_id: 'com.example.app2',
      app_name: 'Sample App 2',
      analysis_date: '2024-01-15T10:30:00Z',
      total_reviews_analyzed: 1203,
      total_reviews_filtered: 297,
      sentiment_distribution: {
        positive: 42.1,
        negative: 31.8,
        neutral: 26.1
      },
      overall_sentiment: 'neutral',
      confidence_metrics: {
        average_confidence: 0.61,
        high_confidence_reviews: 456,
        low_confidence_reviews: 234
      },
      quality_metrics: {
        average_quality_score: 0.68,
        high_quality_reviews: 723,
        low_quality_reviews: 89
      },
      platform_analysis: {
        app_store_weight: 0.38,
        play_store_weight: 0.62,
        app_store_reviews: 457,
        play_store_reviews: 746
      },
      trend_analysis: {
        trend_direction: 'declining',
        anomaly_detected: true,
        confidence_change: -0.12,
        weekly_sentiment_change: {
          positive: -0.15,
          negative: 0.12,
          neutral: 0.03
        }
      },
      flags: {
        manual_review_needed: true,
        anomaly_detected: true,
        low_confidence_batch: true
      },
      language_distribution: {
        'en': 782,
        'es': 241,
        'zh': 134,
        'hi': 31,
        'ar': 15
      }
    }
  };

  const mockTrendData = {
    'com.example.app1': [
      { week: 'Week 1', positive: 65, negative: 20, neutral: 15 },
      { week: 'Week 2', positive: 67, negative: 19, neutral: 14 },
      { week: 'Week 3', positive: 69, negative: 18, neutral: 13 },
      { week: 'Current Week', positive: 68.5, negative: 18.2, neutral: 13.3 }
    ],
    'com.example.app2': [
      { week: 'Week 1', positive: 58, negative: 22, neutral: 20 },
      { week: 'Week 2', positive: 52, negative: 26, neutral: 22 },
      { week: 'Week 3', positive: 48, negative: 28, neutral: 24 },
      { week: 'Current Week', positive: 42.1, negative: 31.8, neutral: 26.1 }
    ]
  };

  const availableApps = [
    { id: 'com.example.app1', name: 'Sample App 1' },
    { id: 'com.example.app2', name: 'Sample App 2' }
  ];

  const loadAnalysisData = async (appId) => {
    if (!appId) return;
    
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setAnalysisData(mockAnalysisData[appId]);
      setTrendData(mockTrendData[appId]);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error loading analysis data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedApp) {
      loadAnalysisData(selectedApp);
    }
  }, [selectedApp]);

  const getTrendIcon = (direction) => {
    switch (direction) {
      case 'improving':
        return <TrendingUp className="w-5 h-5 text-green-500" />;
      case 'declining':
        return <TrendingDown className="w-5 h-5 text-red-500" />;
      case 'volatile':
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      default:
        return <Minus className="w-5 h-5 text-gray-500" />;
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return 'text-green-600 bg-green-100';
      case 'negative':
        return 'text-red-600 bg-red-100';
      case 'neutral':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const pieColors = ['#10B981', '#EF4444', '#F59E0B', '#6B7280'];

  const formatPercentage = (value) => `${value.toFixed(1)}%`;

  const refreshData = () => {
    if (selectedApp) {
      loadAnalysisData(selectedApp);
    }
  };

  if (!analysisData && !loading) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-8 text-center">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Review Analysis Dashboard</h1>
          <p className="text-gray-600 mb-8">Select an app to view its sentiment analysis and trends</p>
          
          <div className="max-w-md mx-auto">
            <select 
              value={selectedApp}
              onChange={(e) => setSelectedApp(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">Choose an app...</option>
              {availableApps.map(app => (
                <option key={app.id} value={app.id}>{app.name}</option>
              ))}
            </select>
          </div>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-8 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading analysis data...</p>
        </div>
      </div>
    );
  }

  const currentWeekData = trendData[trendData.length - 1];
  const previousWeekData = trendData[trendData.length - 2];

  const sentimentPieData = [
    { name: 'Positive', value: analysisData.sentiment_distribution.positive, color: '#10B981' },
    { name: 'Negative', value: analysisData.sentiment_distribution.negative, color: '#EF4444' },
    { name: 'Neutral', value: analysisData.sentiment_distribution.neutral, color: '#F59E0B' }
  ];

  const platformData = [
    { name: 'App Store', reviews: analysisData.platform_analysis.app_store_reviews, weight: analysisData.platform_analysis.app_store_weight },
    { name: 'Play Store', reviews: analysisData.platform_analysis.play_store_reviews, weight: analysisData.platform_analysis.play_store_weight }
  ];

  const languageData = Object.entries(analysisData.language_distribution).map(([lang, count]) => ({
    name: lang.toUpperCase(),
    value: count
  }));

  return (
    <div className="p-6 max-w-7xl mx-auto bg-gray-50 min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-3xl font-bold text-gray-800">Review Analysis Dashboard</h1>
          <div className="flex items-center gap-4">
            <button
              onClick={refreshData}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
            <select 
              value={selectedApp}
              onChange={(e) => setSelectedApp(e.target.value)}
              className="p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">Choose an app...</option>
              {availableApps.map(app => (
                <option key={app.id} value={app.id}>{app.name}</option>
              ))}
            </select>
          </div>
        </div>
        
        <div className="flex items-center gap-4 text-sm text-gray-600">
          <span>Last updated: {lastUpdated.toLocaleString()}</span>
          <span>•</span>
          <span>{analysisData.app_name}</span>
        </div>
      </div>

      {/* Alert Flags */}
      {(analysisData.flags.manual_review_needed || analysisData.flags.anomaly_detected) && (
        <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-5 h-5 text-yellow-600" />
            <h3 className="font-semibold text-yellow-800">Attention Required</h3>
          </div>
          <div className="text-sm text-yellow-700">
            {analysisData.flags.manual_review_needed && (
              <p>• Manual review recommended due to low confidence scores</p>
            )}
            {analysisData.flags.anomaly_detected && (
              <p>• Sentiment anomaly detected in recent reviews</p>
            )}
          </div>
        </div>
      )}

      {/* Main Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        {/* Overall Sentiment */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold text-gray-700">Overall Sentiment</h3>
            {getTrendIcon(analysisData.trend_analysis.trend_direction)}
          </div>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${getSentimentColor(analysisData.overall_sentiment)}`}>
            {analysisData.overall_sentiment.toUpperCase()}
          </div>
          <div className="mt-2 text-2xl font-bold text-gray-800">
            {formatPercentage(analysisData.sentiment_distribution.positive)}
          </div>
          <p className="text-sm text-gray-600">Positive reviews</p>
        </div>

        {/* Reviews Analyzed */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold text-gray-700">Reviews Processed</h3>
            <Filter className="w-5 h-5 text-blue-500" />
          </div>
          <div className="text-2xl font-bold text-gray-800">
            {analysisData.total_reviews_analyzed.toLocaleString()}
          </div>
          <p className="text-sm text-gray-600">
            {analysisData.total_reviews_filtered} filtered out
          </p>
          <div className="mt-2 text-xs text-gray-500">
            {formatPercentage((analysisData.total_reviews_analyzed / (analysisData.total_reviews_analyzed + analysisData.total_reviews_filtered)) * 100)} quality rate
          </div>
        </div>

        {/* Confidence Score */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold text-gray-700">Confidence</h3>
            {analysisData.confidence_metrics.average_confidence >= 0.65 ? 
              <CheckCircle className="w-5 h-5 text-green-500" /> : 
              <AlertTriangle className="w-5 h-5 text-yellow-500" />
            }
          </div>
          <div className="text-2xl font-bold text-gray-800">
            {formatPercentage(analysisData.confidence_metrics.average_confidence * 100)}
          </div>
          <p className="text-sm text-gray-600">Average confidence</p>
          {analysisData.confidence_metrics.average_confidence < 0.65 && (
            <div className="mt-2 text-xs text-red-600 font-medium">
              Manual review recommended
            </div>
          )}
        </div>

        {/* Week vs Week Change */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold text-gray-700">Weekly Change</h3>
            <Clock className="w-5 h-5 text-purple-500" />
          </div>
          <div className="text-2xl font-bold text-gray-800">
            {analysisData.trend_analysis.weekly_sentiment_change.positive >= 0 ? '+' : ''}
            {formatPercentage(analysisData.trend_analysis.weekly_sentiment_change.positive)}
          </div>
          <p className="text-sm text-gray-600">Positive sentiment change</p>
          <div className="mt-2 text-xs text-gray-500">
            vs. previous week
          </div>
        </div>
      </div>

      {/* Platform Analysis & Sentiment Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Platform Weight Distribution */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Platform Distribution</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-4 h-4 bg-blue-500 rounded"></div>
                <span className="text-gray-700">App Store</span>
              </div>
              <div className="text-right">
                <div className="font-semibold">{analysisData.platform_analysis.app_store_reviews} reviews</div>
                <div className="text-sm text-gray-600">{formatPercentage(analysisData.platform_analysis.app_store_weight * 100)} weight</div>
              </div>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full" 
                style={{ width: `${analysisData.platform_analysis.app_store_weight * 100}%` }}
              ></div>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-4 h-4 bg-green-500 rounded"></div>
                <span className="text-gray-700">Play Store</span>
              </div>
              <div className="text-right">
                <div className="font-semibold">{analysisData.platform_analysis.play_store_reviews} reviews</div>
                <div className="text-sm text-gray-600">{formatPercentage(analysisData.platform_analysis.play_store_weight * 100)} weight</div>
              </div>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-green-500 h-2 rounded-full" 
                style={{ width: `${analysisData.platform_analysis.play_store_weight * 100}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* Sentiment Distribution Pie Chart */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Sentiment Distribution</h3>
          <div className="flex items-center justify-center">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={sentimentPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {sentimentPieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => formatPercentage(value)} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-3 gap-4 mt-4">
            {sentimentPieData.map((item, index) => (
              <div key={index} className="text-center">
                <div className="flex items-center justify-center gap-2 mb-1">
                  <div className={`w-3 h-3 rounded-full`} style={{ backgroundColor: item.color }}></div>
                  <span className="text-sm text-gray-600">{item.name}</span>
                </div>
                <div className="font-semibold text-gray-800">{formatPercentage(item.value)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Week vs Week Comparison */}
      <div className="bg-white p-6 rounded-lg shadow-lg mb-8">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Weekly Trend Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-700 mb-2">Current Week</h4>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-green-600">Positive:</span>
                <span className="font-semibold">{formatPercentage(currentWeekData.positive)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-red-600">Negative:</span>
                <span className="font-semibold">{formatPercentage(currentWeekData.negative)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-yellow-600">Neutral:</span>
                <span className="font-semibold">{formatPercentage(currentWeekData.neutral)}</span>
              </div>
            </div>
          </div>
          
          <div className="p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-700 mb-2">Previous Week</h4>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-green-600">Positive:</span>
                <span className="font-semibold">{formatPercentage(previousWeekData.positive)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-red-600">Negative:</span>
                <span className="font-semibold">{formatPercentage(previousWeekData.negative)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-yellow-600">Neutral:</span>
                <span className="font-semibold">{formatPercentage(previousWeekData.neutral)}</span>
              </div>
            </div>
          </div>
        </div>
        
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="week" />
            <YAxis />
            <Tooltip formatter={(value) => formatPercentage(value)} />
            <Line type="monotone" dataKey="positive" stroke="#10B981" strokeWidth={2} />
            <Line type="monotone" dataKey="negative" stroke="#EF4444" strokeWidth={2} />
            <Line type="monotone" dataKey="neutral" stroke="#F59E0B" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Quality & Language Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Review Quality Metrics */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Review Quality Analysis</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Average Quality Score</span>
              <span className="font-semibold text-lg">{formatPercentage(analysisData.quality_metrics.average_quality_score * 100)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">High Quality Reviews</span>
              <span className="font-semibold">{analysisData.quality_metrics.high_quality_reviews}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Low Quality Reviews</span>
              <span className="font-semibold text-red-600">{analysisData.quality_metrics.low_quality_reviews}</span>
            </div>
            <div className="mt-4 p-3 bg-blue-50 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Filter className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-medium text-blue-800">Filtering Status</span>
              </div>
              <p className="text-sm text-blue-700">
                {analysisData.total_reviews_filtered} low-quality reviews filtered out (one-word responses, duplicates, spam)
              </p>
            </div>
          </div>
        </div>

        {/* Language Distribution */}
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Language Distribution</h3>
          <div className="space-y-3">
            {Object.entries(analysisData.language_distribution).map(([lang, count]) => {
              const percentage = (count / analysisData.total_reviews_analyzed) * 100;
              const langNames = {
                'en': 'English',
                'es': 'Spanish', 
                'zh': 'Chinese',
                'hi': 'Hindi',
                'ar': 'Arabic'
              };
              return (
                <div key={lang} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="text-gray-700 font-medium">{langNames[lang] || lang.toUpperCase()}</span>
                    <span className="text-sm text-gray-500">({count} reviews)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-20 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-purple-500 h-2 rounded-full" 
                        style={{ width: `${percentage}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium">{formatPercentage(percentage)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Anomaly Detection & Recommendations */}
      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Analysis Summary & Recommendations</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-700 mb-3">Trend Analysis</h4>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                {getTrendIcon(analysisData.trend_analysis.trend_direction)}
                <span className="text-gray-700">
                  Sentiment is {analysisData.trend_analysis.trend_direction}
                </span>
              </div>
              <div className="flex items-center gap-2">
                {analysisData.trend_analysis.anomaly_detected ? 
                  <AlertTriangle className="w-4 h-4 text-red-500" /> : 
                  <CheckCircle className="w-4 h-4 text-green-500" />
                }
                <span className="text-gray-700">
                  {analysisData.trend_analysis.anomaly_detected ? 'Anomaly detected' : 'No anomalies detected'}
                </span>
              </div>
              <div className="text-sm text-gray-600">
                Confidence change: {analysisData.trend_analysis.confidence_change >= 0 ? '+' : ''}{formatPercentage(analysisData.trend_analysis.confidence_change * 100)}
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-medium text-gray-700 mb-3">Action Items</h4>
            <div className="space-y-2">
              {analysisData.flags.manual_review_needed && (
                <div className="flex items-center gap-2 text-orange-600">
                  <AlertTriangle className="w-4 h-4" />
                  <span className="text-sm">Manual review recommended</span>
                </div>
              )}
              {analysisData.flags.anomaly_detected && (
                <div className="flex items-center gap-2 text-red-600">
                  <AlertTriangle className="w-4 h-4" />
                  <span className="text-sm">Investigate sentiment anomaly</span>
                </div>
              )}
              {analysisData.flags.low_confidence_batch && (
                <div className="flex items-center gap-2 text-yellow-600">
                  <AlertTriangle className="w-4 h-4" />
                  <span className="text-sm">Low confidence batch - verify results</span>
                </div>
              )}
              {!analysisData.flags.manual_review_needed && !analysisData.flags.anomaly_detected && !analysisData.flags.low_confidence_batch && (
                <div className="flex items-center gap-2 text-green-600">
                  <CheckCircle className="w-4 h-4" />
                  <span className="text-sm">All systems normal</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReviewAnalysisDashboard;
