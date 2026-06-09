import { Toaster } from 'react-hot-toast';
import InterviewScreen from '@/pages/Interview/InterviewScreen';

function App() {
  return (
    <>
      <Toaster position="top-right" />
      <InterviewScreen />
    </>
  );
}

export default App;
