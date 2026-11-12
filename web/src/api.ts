export async function apiGet<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

export async function apiPost<T>(url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

export async function apiUpload<T>(url: string, file: File): Promise<T> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(url, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

export async function apiUploadMultiple<T>(url: string, files: File[]): Promise<T> {
  const form = new FormData();
  files.forEach((f) => form.append('files', f));
  const res = await fetch(url, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

export function referenceContactSheetUrl(referenceId: string): string {
  return `/api/references/${encodeURIComponent(referenceId)}/contact-sheet`;
}

export function workspaceFileUrl(projectId: string, relPath: string): string {
  return `/api/projects/${encodeURIComponent(projectId)}/files/${relPath.split('/').map(encodeURIComponent).join('/')}`;
}
