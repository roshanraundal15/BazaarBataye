
import { useState } from 'react';

const steps = [
  {
    number: 1,
    title: "Access the Tool",
    description: "Click the 🌾 Predict Now button to open the tool.",
    icon: "🔎"
  },
  {
    number: 2,
    title: "Select Your Crop",
    description: "Select your crop, location, and variety.",
    icon: "🌱"
  },
  {
    number: 3,
    title: "Enter Details",
    description: "Enter your expected yield (in quintals) and your investment amount.",
    icon: "📝"
  },
  {
    number: 4,
    title: "Get Forecast",
    description: "Get the forecasted market price.",
    icon: "📊"
  },
  {
    number: 5,
    title: "Review Results",
    description: "See your estimated profit and AI advice on whether to sell or hold.",
    icon: "💡"
  }
];

const UserGuide = () => {
  const [activeTab, setActiveTab] = useState<number>(0);

  return (
    <section className="py-16 bg-secondary/50" id="user-guide">
      <div className="container mx-auto px-4 md:px-6">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-ag-earth mb-4">👨‍🌾 How to Use BazaarBataye</h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Follow these simple steps to get accurate price forecasts and make informed decisions
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          {/* Mobile Step Tabs */}
          <div className="md:hidden flex overflow-x-auto gap-2 pb-4 mb-4">
            {steps.map((step, index) => (
              <button
                key={index}
                onClick={() => setActiveTab(index)}
                className={`flex-shrink-0 h-10 w-10 rounded-full flex items-center justify-center ${
                  activeTab === index 
                    ? 'bg-ag-green text-white' 
                    : 'bg-gray-100 text-gray-500'
                }`}
              >
                {step.number}
              </button>
            ))}
          </div>

          {/* Desktop Steps */}
          <div className="hidden md:grid grid-cols-5 gap-4 mb-8">
            {steps.map((step, index) => (
              <div 
                key={index}
                className={`relative cursor-pointer group ${
                  index < steps.length - 1 ? 'step-connector' : ''
                }`}
                onClick={() => setActiveTab(index)}
              >
                <div className={`
                  flex flex-col items-center ${activeTab === index ? 'scale-110' : 'opacity-60'}
                  transition-all duration-300
                `}>
                  <div className={`
                    h-16 w-16 rounded-full flex items-center justify-center text-2xl mb-3
                    ${activeTab === index ? 'bg-ag-green text-white' : 'bg-gray-100 text-gray-600'}
                    transition-colors duration-300
                  `}>
                    {step.icon}
                  </div>
                  <span className={`text-sm font-medium ${activeTab === index ? 'text-ag-earth' : 'text-gray-500'}`}>
                    Step {step.number}
                  </span>
                </div>
                
                {/* Connector line */}
                {index < steps.length - 1 && (
                  <div className="absolute top-8 left-full w-full h-0.5 bg-gray-200 -z-10 pointer-events-none">
                    <div className={`h-full bg-ag-green transition-all duration-500 ${
                      activeTab > index ? 'w-full' : 'w-0'
                    }`}></div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Content for active step */}
          <div className="bg-white rounded-xl shadow-lg p-6 md:p-8 animate-fade-in">
            <div className="flex items-start gap-4">
              <div className="hidden md:flex h-14 w-14 flex-shrink-0 rounded-full bg-ag-green/10 items-center justify-center text-2xl text-ag-green">
                {steps[activeTab].icon}
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-ag-earth flex items-center gap-2">
                  <span className="md:hidden">{steps[activeTab].icon}</span>
                  <span>Step {steps[activeTab].number}: {steps[activeTab].title}</span>
                </h3>
                <p className="mt-2 text-gray-600">{steps[activeTab].description}</p>
                
                {/* Additional content for each step */}
                <div className="mt-6">
                  {activeTab === 0 && (
                    <div className="bg-ag-green/5 p-4 rounded-lg">
                      <p className="text-ag-green-dark">
                        Look for the 🌾 Predict Now button in the top-right corner of your screen.
                      </p>
                    </div>
                  )}
                  
                  {activeTab === 4 && (
                    <div className="bg-ag-gold/5 p-4 rounded-lg">
                      <p className="text-ag-gold-dark">
                        <strong>Pro tip:</strong> Our AI analyzes historical trends, weather patterns, and market dynamics to provide accurate forecasts.
                      </p>
                    </div>
                  )}
                </div>
                
                {/* Navigation buttons */}
                <div className="flex justify-between mt-8">
                  <button
                    onClick={() => setActiveTab(prev => Math.max(0, prev - 1))}
                    disabled={activeTab === 0}
                    className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
                      activeTab === 0 
                        ? 'text-gray-400 cursor-not-allowed' 
                        : 'text-ag-earth hover:bg-ag-green/10'
                    }`}
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                    Previous
                  </button>
                  
                  <button
                    onClick={() => setActiveTab(prev => Math.min(steps.length - 1, prev + 1))}
                    disabled={activeTab === steps.length - 1}
                    className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
                      activeTab === steps.length - 1 
                        ? 'text-gray-400 cursor-not-allowed' 
                        : 'text-ag-green hover:bg-ag-green/10'
                    }`}
                  >
                    Next
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default UserGuide;
