import { Hono } from "hono";

const app = new Hono();

app.get("/", (c) => {
    return c.text("Hello, World!");
});

app.get("/:name", (c) => {
    const name = c.req.param("name");
    return c.text(`Hello, ${name}!`);
});
export default app;
