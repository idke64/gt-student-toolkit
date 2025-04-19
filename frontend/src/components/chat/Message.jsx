export default function Message({ content }) {
  return (
    <div className="p-3 rounded-lg max-w-[75%] bg-gray-50 text-black border-2 ml-auto">
      {content}
    </div>
  );
}
