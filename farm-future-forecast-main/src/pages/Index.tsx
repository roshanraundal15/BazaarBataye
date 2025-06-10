
import { Brain, DollarSign, Search, TrendingUp } from 'lucide-react';
import { useEffect } from 'react';
import FeatureCard from '../components/FeatureCard';
import Footer from '../components/Footer';
import HeroSection from '../components/HeroSection';
import NavBar from '../components/NavBar';
import UserGuide from '../components/UserGuide';

const Index = () => {
  // Intersection Observer setup for scroll animations
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
          }
        });
      },
      {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
      }
    );

    // Observe all elements with animate-on-scroll class
    document.querySelectorAll('.animate-on-scroll').forEach(element => {
      observer.observe(element);
    });

    return () => {
      document.querySelectorAll('.animate-on-scroll').forEach(element => {
        observer.unobserve(element);
      });
    };
  }, []);

  const features = [
    {
      title: "Forecast Crop Prices",
      description: "Use advanced AI to predict future market prices for your crops with remarkable accuracy.",
      icon: <Search className="w-6 h-6 text-ag-green" />,
      delay: 0
    },
    {
      title: "Estimate Profit",
      description: "Calculate potential profits based on predicted prices, yield, and your investment costs.",
      icon: <DollarSign className="w-6 h-6 text-ag-green" />,
      delay: 150
    },
    {
      title: "Make Better Selling Decisions",
      description: "Get data-driven recommendations on whether to sell now or hold for better prices.",
      icon: <TrendingUp className="w-6 h-6 text-ag-green" />,
      delay: 300
    },
    {
      title: "Powered by AI",
      description: "Benefit from machine learning models that continuously improve prediction accuracy.",
      icon: <Brain className="w-6 h-6 text-ag-green" />,
      delay: 450
    }
  ];

  return (
    <div className="min-h-screen flex flex-col">
      <NavBar />
      
      <main className="flex-grow">
        <HeroSection />
        
        {/* Features Section */}
        <section className="py-16 md:py-24">
          <div className="container mx-auto px-4 md:px-6">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-ag-earth mb-4">
                Advanced Features for Modern Farmers
              </h2>
              <p className="text-gray-600 max-w-2xl mx-auto">
                Get ahead of market fluctuations with our powerful AI-driven tools designed specifically for agricultural businesses.
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {features.map((feature, index) => (
                <FeatureCard
                  key={index}
                  title={feature.title}
                  description={feature.description}
                  icon={feature.icon}
                  delay={feature.delay}
                />
              ))}
            </div>
          </div>
        </section>
        
        {/* User Guide Section */}
        <UserGuide />
        
        {/* CTA Section */}
        <section className="py-16 md:py-20 bg-gradient-to-br from-ag-green to-ag-green-dark text-white">
          <div className="container mx-auto px-4 md:px-6 text-center">
            <h2 className="text-3xl md:text-4xl font-bold mb-6">Ready to Optimize Your Crop Sales?</h2>
            <p className="text-white/90 max-w-2xl mx-auto mb-8 text-lg">
              Join thousands of farmers already using BazaarBataye to maximize their profits with data-driven market insights.
            </p>
            
            <a 
              href="http://localhost:8501" 
              className="inline-flex items-center px-8 py-3 bg-white text-ag-green-dark rounded-full 
                         font-medium shadow-lg hover:shadow-xl hover:bg-ag-gold-light transition-all duration-300"
            >
              <span className="mr-2">🌾</span>
              <span>Start Predicting Now</span>
              <svg className="w-5 h-5 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            </a>
            
            <p className="mt-6 text-white/80 text-sm">No credit card required. Start making smarter decisions today.</p>
          </div>
        </section>
      </main>
      
      <Footer />
    </div>
  );
};

export default Index;
