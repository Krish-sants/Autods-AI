import UploadDropzone from "@/components/UploadDropzone";

export default function Home() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-8 px-6 py-20">
      <div className="text-center">
        <h1 className="text-3xl font-semibold">AutoDS-AI</h1>
        <p className="mt-2 text-zinc-500">
          Upload a dataset and let an autonomous data scientist handle the rest.
        </p>
      </div>
      <UploadDropzone />
    </div>
  );
}
