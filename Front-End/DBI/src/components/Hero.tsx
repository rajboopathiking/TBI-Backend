import image1 from "../../public/Images/image-1.jpg"

export default function Hero() {
  return (
    <div className='flex sm:block items-center justify-center' >
      <div className='m-10'>
        <img className='h-[500px] sm:w-full rounded-xl w-[30vw] z-1' src={image1} alt="" />
      </div>
      <div>
        <p className="font-bold p-5 w-[500px] sm:w-full  md:w-[300px]" id='hero_animation'>Access the website interface to seamlessly input critical physiological parameters for pediatric severe TBI patients. Receive instant alerts generated by the predictive model deployed in the cloud, facilitating proactive healthcare interventions.</p>
      </div>
    </div>
  )
}