import { ShieldCheck } from 'lucide-react';

const HeroSection = () => {
  const handlePredictNow = () => {
    window.location.href = 'https://agri-predictor-gdg.streamlit.app/';
  };

  const handleQualityCheck = () => {
    window.location.href = 'https://image-detection-gdg.streamlit.app/';
  };

  return (
    <section className="relative py-16 md:py-24 overflow-hidden">
      <div className="absolute inset-0 bg-hero-gradient opacity-50 pointer-events-none"></div>
      
      <div className="container mx-auto px-4 md:px-6">
        <div className="flex flex-col md:flex-row items-center gap-12">
          <div className="md:w-1/2 space-y-6 z-10">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-ag-earth hero-glow">
            "Bazaar kya bolega, ab pehle se pata."
            </h1>
            
            <p className="text-lg md:text-xl text-gray-700 max-w-xl">
              Get AI-powered crop price forecasts to make smart selling or holding decisions.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 mt-4">
              <button 
                onClick={handlePredictNow}
                className="cta-button bg-gradient-to-r from-ag-green to-ag-green-dark hover:from-ag-green-dark hover:to-ag-green 
                          text-white font-medium py-3 px-8 rounded-full shadow-lg hover:shadow-xl transition-all duration-300
                          flex items-center gap-2 group"
              >
                <span className="mr-2">🌾</span>
                <span>Start Predicting</span>
                <svg 
                  className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </button>
              
              <button 
                onClick={handleQualityCheck}
                className="cta-button bg-gradient-to-r from-ag-gold to-ag-gold-light hover:from-ag-gold-light hover:to-ag-gold 
                          text-white font-medium py-3 px-8 rounded-full shadow-lg hover:shadow-xl transition-all duration-300
                          flex items-center gap-2 group"
              >
                <ShieldCheck className="w-5 h-5" />
                <span>Quality Check</span>
                <svg 
                  className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </button>
            </div>
          </div>
          
          <div className="md:w-1/2 flex justify-center">
            <div className="relative w-full max-w-lg h-80 md:h-96">
              <div className="absolute top-0 -left-4 w-40 h-40 bg-ag-gold/20 rounded-full mix-blend-multiply filter blur-2xl opacity-70 animate-float"></div>
              <div className="absolute -bottom-8 left-20 w-36 h-36 bg-ag-green/20 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-float" style={{animationDelay: '2s'}}></div>
              <div className="absolute top-20 -right-4 w-32 h-32 bg-ag-sky/20 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-float" style={{animationDelay: '4s'}}></div>
              
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="relative w-64 h-64 md:w-80 md:h-80 bg-white rounded-full shadow-2xl overflow-hidden border-8 border-white">
                  <div className="absolute inset-0 bg-gradient-to-br from-ag-green-light/30 to-ag-gold-light/30"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-8xl animate-pulse">🌾</div>
                  </div>
                  <div className="absolute bottom-6 inset-x-0 text-center text-ag-earth text-sm font-semibold">AI-Powered Insights</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
