import type { Config } from "@netlify/functions";

export default async function health() {
  return Response.json({ status: "ok" });
}

export const config: Config = {
  path: "/api/health",
};

