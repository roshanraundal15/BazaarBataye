const Footer = () => {
  return (
    <footer className="bg-[#1B1F23] py-12 text-white">
      <div className="container mx-auto px-4 md:px-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          
          {/* GitHub Links */}
          <div>
            <h4 className="text-lg font-semibold mb-4 text-yellow-300">GitHub</h4>
            <ul className="space-y-2">
              <li>
                <a
                  href="https://github.com/roshanraundal15"
                  className="text-white/80 hover:text-white transition-colors"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Roshan Raundal
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/karanpanchallll"
                  className="text-white/80 hover:text-white transition-colors"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Karan Panchal
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/mayursapkal41"
                  className="text-white/80 hover:text-white transition-colors"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Mayur Sapkal
                </a>
              </li>
            </ul>
          </div>

          {/* Contact Information */}
          <div>
            <h4 className="text-lg font-semibold mb-4 text-yellow-300">Contact</h4>
            <address className="not-italic space-y-2">
              <p className="text-white/80">raundal2001@gmail.com</p>
              <p className="text-white/80">panchalkaran@gmail.com</p>
              <p className="text-white/80">mayursapkal41@gmail.com</p>
            </address>
          </div>

          {/* YouTube Video */}
          <div>
            <h4 className="text-lg font-semibold mb-4 text-yellow-300">Demo Video</h4>
            <div className="relative pt-[56.25%] w-full overflow-hidden rounded-xl">
              <iframe
                className="absolute top-0 left-0 w-full h-full"
                src="https://www.youtube.com/embed/yYJNuWuqjG8"
                title="Bazaar Bataye Demo"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              ></iframe>
            </div>
          </div>
        </div>

        {/* Footer Description */}
        <div className="mt-12 text-center">
          <h3 className="text-xl font-bold mb-2 text-yellow-300">BazaarBataye</h3>
          <p className="text-white/80">
            Empowering farmers with AI-powered crop price predictions and market insights.
          </p>
        </div>

        {/* Bottom Footer */}
        <div className="mt-8 pt-8 border-t border-white/10 text-center text-white/60">
          <p>&copy; {new Date().getFullYear()} BazaarBataye: Agriculture Market Price Predictor. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
